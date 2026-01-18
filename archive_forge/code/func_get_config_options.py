import atexit
import errno
import os
import pathlib
import re
import sys
import tempfile
import ast
import warnings
import shutil
from io import StringIO
from docutils.parsers.rst import directives
from docutils.parsers.rst import Directive
from sphinx.util import logging
from traitlets.config import Config
from IPython import InteractiveShell
from IPython.core.profiledir import ProfileDir
def get_config_options(self):
    config = self.state.document.settings.env.config
    savefig_dir = config.ipython_savefig_dir
    source_dir = self.state.document.settings.env.srcdir
    savefig_dir = os.path.join(source_dir, savefig_dir)
    rgxin = config.ipython_rgxin
    rgxout = config.ipython_rgxout
    warning_is_error = config.ipython_warning_is_error
    promptin = config.ipython_promptin
    promptout = config.ipython_promptout
    mplbackend = config.ipython_mplbackend
    exec_lines = config.ipython_execlines
    hold_count = config.ipython_holdcount
    return (savefig_dir, source_dir, rgxin, rgxout, promptin, promptout, mplbackend, exec_lines, hold_count, warning_is_error)