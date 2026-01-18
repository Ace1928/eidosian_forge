import inspect
import linecache
import sys
import re
import os
from IPython import get_ipython
from contextlib import contextmanager
from IPython.utils import PyColorize
from IPython.utils import coloransi, py3compat
from IPython.core.excolors import exception_colors
from pdb import Pdb as OldPdb
def do_context(self, context):
    """context number_of_lines
        Set the number of lines of source code to show when displaying
        stacktrace information.
        """
    try:
        new_context = int(context)
        if new_context <= 0:
            raise ValueError()
        self.context = new_context
    except ValueError:
        self.error("The 'context' command requires a positive integer argument.")