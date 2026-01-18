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
def convert_single_notebook(self, notebook_filename, input_buffer=None):
    """Convert a single notebook.

        Performs the following steps:

            1. Initialize notebook resources
            2. Export the notebook to a particular format
            3. Write the exported notebook to file
            4. (Maybe) postprocess the written file

        Parameters
        ----------
        notebook_filename : str
        input_buffer :
            If input_buffer is not None, conversion is done and the buffer is
            used as source into a file basenamed by the notebook_filename
            argument.
        """
    if input_buffer is None:
        self.log.info('Converting notebook %s to %s', notebook_filename, self.export_format)
    else:
        self.log.info('Converting notebook into %s', self.export_format)
    resources = self.init_single_notebook_resources(notebook_filename)
    output, resources = self.export_single_notebook(notebook_filename, resources, input_buffer=input_buffer)
    write_results = self.write_single_notebook(output, resources)
    self.postprocess_single_notebook(write_results)