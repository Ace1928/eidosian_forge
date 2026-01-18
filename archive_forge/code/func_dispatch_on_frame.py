from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
def dispatch_on_frame(c_command, python_command=None):

    def decorator(function):

        @functools.wraps(function)
        def wrapper(self, *args, **kwargs):
            is_cy = self.is_cython_function()
            is_py = self.is_python_function()
            if is_cy or (is_py and (not python_command)):
                function(self, *args, **kwargs)
            elif is_py:
                gdb.execute(python_command)
            elif self.is_relevant_function():
                gdb.execute(c_command)
            else:
                raise gdb.GdbError('Not a function cygdb knows about. Use the normal GDB commands instead.')
        return wrapper
    return decorator