import argparse
import bdb
import locale
import multiprocessing
import os
import pdb
import sys
import traceback
from os import path
from typing import Any, List, Optional, TextIO
from docutils.utils import SystemMessage
import sphinx.locale
from sphinx import __display_version__, package_dir
from sphinx.application import Sphinx
from sphinx.errors import SphinxError
from sphinx.locale import __
from sphinx.util import Tee, format_exception_cut_frames, save_traceback
from sphinx.util.console import color_terminal, nocolor, red, terminal_safe  # type: ignore
from sphinx.util.docutils import docutils_namespace, patch_docutils
from sphinx.util.osutil import abspath, ensuredir
def jobs_argument(value: str) -> int:
    """
    Special type to handle 'auto' flags passed to 'sphinx-build' via -j flag. Can
    be expanded to handle other special scaling requests, such as setting job count
    to cpu_count.
    """
    if value == 'auto':
        return multiprocessing.cpu_count()
    else:
        jobs = int(value)
        if jobs <= 0:
            raise argparse.ArgumentTypeError(__('job number should be a positive number'))
        else:
            return jobs