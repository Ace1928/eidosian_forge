from __future__ import annotations
import ast
import html
import json
import logging
import os
import pathlib
import re
import sys
import traceback
import urllib.parse as urlparse
from contextlib import contextmanager
from types import ModuleType
from typing import IO, Any, Callable
import bokeh.command.util
from bokeh.application.handlers.code import CodeHandler
from bokeh.application.handlers.code_runner import CodeRunner
from bokeh.application.handlers.handler import Handler, handle_exception
from bokeh.core.types import PathLike
from bokeh.document import Document
from bokeh.io.doc import curdoc, patch_curdoc, set_curdoc as bk_set_curdoc
from bokeh.util.dependencies import import_required
from ..config import config
from .mime_render import MIME_RENDERERS
from .profile import profile_ctx
from .reload import record_modules
from .state import state
def parse_notebook(filename: str | os.PathLike | IO, preamble: list[str] | None=None):
    """
    Parses a notebook on disk and returns a script.

    Arguments
    ---------
    filename: str | os.PathLike
      The notebook file to parse.
    preamble: list[str]
      Any lines of code to prepend to the parsed code output.

    Returns
    -------
    nb: nbformat.NotebookNode
      nbformat dictionary-like representation of the notebook
    code: str
      The parsed and converted script
    cell_layouts: dict
      Dictionary containing the layout and positioning of cells.
    """
    nbconvert = import_required('nbconvert', 'The Panel notebook application handler requires nbconvert to be installed.')
    nbformat = import_required('nbformat', 'The Panel notebook application handler requires Jupyter Notebook to be installed.')

    class StripMagicsProcessor(nbconvert.preprocessors.Preprocessor):
        """
        Preprocessor to convert notebooks to Python source while stripping
        out all magics (i.e IPython specific syntax).
        """
        _magic_pattern = re.compile('^\\s*(?P<magic>%%\\w\\w+)($|(\\s+))')

        def strip_magics(self, source: str) -> str:
            """
            Given the source of a cell, filter out all cell and line magics.
            """
            filtered: list[str] = []
            for line in source.splitlines():
                match = self._magic_pattern.match(line)
                if match is None:
                    filtered.append(line)
                else:
                    msg = 'Stripping out IPython magic {magic} in code cell {cell}'
                    message = msg.format(cell=self._cell_counter, magic=match.group('magic'))
                    log.warning(message)
            return '\n'.join(filtered)

        def preprocess_cell(self, cell, resources, index):
            if cell['cell_type'] == 'code':
                self._cell_counter += 1
                cell['source'] = self.strip_magics(cell['source'])
            return (cell, resources)

        def __call__(self, nb, resources):
            self._cell_counter = 0
            return self.preprocess(nb, resources)
    preprocessors = [StripMagicsProcessor()]
    nb = nbformat.read(filename, nbformat.NO_CONVERT)
    exporter = nbconvert.NotebookExporter()
    for preprocessor in preprocessors:
        exporter.register_preprocessor(preprocessor)
    nb_string, _ = exporter.from_notebook_node(nb)
    nb = nbformat.reads(nb_string, 4)
    nb = nbformat.v4.upgrade(nb)
    cell_layouts = {}
    code = list(preamble or [])
    for cell in nb['cells']:
        cell_id = cell['id']
        cell_layouts[cell_id] = cell['metadata'].get('panel-layout', {})
        if cell['cell_type'] == 'code':
            cell_code = capture_code_cell(cell)
            code += cell_code
        elif cell['cell_type'] == 'markdown':
            md = ''.join(cell['source']).replace('"', '\\"')
            code.append(f'_pn__state._cell_outputs[{cell_id!r}].append("""{md}""")')
    code = '\n'.join(code)
    return (nb, code, cell_layouts)