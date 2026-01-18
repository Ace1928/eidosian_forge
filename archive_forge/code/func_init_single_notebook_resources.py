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
def init_single_notebook_resources(self, notebook_filename):
    """Step 1: Initialize resources

        This initializes the resources dictionary for a single notebook.

        Returns
        -------
        dict
            resources dictionary for a single notebook that MUST include the following keys:
                - config_dir: the location of the Jupyter config directory
                - unique_key: the notebook name
                - output_files_dir: a directory where output files (not
                  including the notebook itself) should be saved
        """
    notebook_name = self._notebook_filename_to_name(notebook_filename)
    self.log.debug("Notebook name is '%s'", notebook_name)
    resources = {}
    resources['config_dir'] = self.config_dir
    resources['unique_key'] = notebook_name
    output_files_dir = self.output_files_dir.format(notebook_name=notebook_name)
    resources['output_files_dir'] = output_files_dir
    return resources