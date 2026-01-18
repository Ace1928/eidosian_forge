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
def decorate_fn_with_doc(new_fn, old_fn, additional_text=''):
    """Make new_fn have old_fn's doc string. This is particularly useful
    for the ``do_...`` commands that hook into the help system.
    Adapted from from a comp.lang.python posting
    by Duncan Booth."""

    def wrapper(*args, **kw):
        return new_fn(*args, **kw)
    if old_fn.__doc__:
        wrapper.__doc__ = strip_indentation(old_fn.__doc__) + additional_text
    return wrapper