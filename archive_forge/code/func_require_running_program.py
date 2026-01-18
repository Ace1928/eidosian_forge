from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
def require_running_program(function):

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            gdb.selected_frame()
        except RuntimeError:
            raise gdb.GdbError('No frame is currently selected.')
        return function(*args, **kwargs)
    return wrapper