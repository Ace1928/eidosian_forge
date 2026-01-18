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
def document_flag_help(self):
    """
        Return a string containing descriptions of all the flags.
        """
    flags = 'The following flags are defined:\n\n'
    for flag, (cfg, fhelp) in self.flags.items():
        flags += f'{flag}\n'
        flags += indent(fill(fhelp, 80)) + '\n\n'
        flags += indent(fill('Long Form: ' + str(cfg), 80)) + '\n\n'
    return flags