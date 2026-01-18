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
def init_notebooks(self):
    """Construct the list of notebooks.

        If notebooks are passed on the command-line,
        they override (rather than add) notebooks specified in config files.
        Glob each notebook to replace notebook patterns with filenames.
        """
    patterns = self.extra_args if self.extra_args else self.notebooks
    filenames = []
    for pattern in patterns:
        globbed_files = glob.glob(pattern, recursive=self.recursive_glob)
        globbed_files.extend(glob.glob(pattern + '.ipynb', recursive=self.recursive_glob))
        if not globbed_files:
            self.log.warning('pattern %r matched no files', pattern)
        for filename in globbed_files:
            if filename not in filenames:
                filenames.append(filename)
    self.notebooks = filenames