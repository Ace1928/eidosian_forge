from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
def gdb_function_value_to_unicode(function):

    @functools.wraps(function)
    def wrapper(self, string, *args, **kwargs):
        if isinstance(string, gdb.Value):
            string = string.string()
        return function(self, string, *args, **kwargs)
    return wrapper