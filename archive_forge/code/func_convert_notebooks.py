from __future__ import annotations
import asyncio
import glob
import logging
import os
import sys
import typing as t
from textwrap import dedent, fill
from jupyter_core.application import JupyterApp, base_aliases, base_flags
from traitlets import Bool, DottedObjectName, Instance, List, Type, Unicode, default, observe
from traitlets.config import Configurable, catch_config_error
from traitlets.utils.importstring import import_item
from nbconvert import __version__, exporters, postprocessors, preprocessors, writers
from nbconvert.utils.text import indent
from .exporters.base import get_export_names, get_exporter
from .utils.base import NbConvertBase
from .utils.exceptions import ConversionException
from .utils.io import unicode_stdin_stream
def convert_notebooks(self):
    """Convert the notebooks in the self.notebooks traitlet"""
    if len(self.notebooks) == 0 and (not self.from_stdin):
        self.print_help()
        sys.exit(-1)
    if not self.export_format:
        msg = f"Please specify an output format with '--to <format>'.\nThe following formats are available: {get_export_names()}"
        raise ValueError(msg)
    cls = get_exporter(self.export_format)
    self.exporter = cls(config=self.config)
    if getattr(self.exporter, 'file_extension', False):
        base, ext = os.path.splitext(self.output_base)
        if ext == self.exporter.file_extension:
            self.output_base = base
    if not self.from_stdin:
        for notebook_filename in self.notebooks:
            self.convert_single_notebook(notebook_filename)
    else:
        input_buffer = unicode_stdin_stream()
        self.convert_single_notebook('notebook.ipynb', input_buffer=input_buffer)
        input_buffer.close()