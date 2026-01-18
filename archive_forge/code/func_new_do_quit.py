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
def new_do_quit(self, arg):
    if hasattr(self, 'old_all_completions'):
        self.shell.Completer.all_completions = self.old_all_completions
    return OldPdb.do_quit(self, arg)