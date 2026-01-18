from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
def print_gdb_value(self, name, value, max_name_length=None, prefix=''):
    if libpython.pretty_printer_lookup(value):
        typename = ''
    else:
        typename = '(%s) ' % (value.type,)
    if max_name_length is None:
        print('%s%s = %s%s' % (prefix, name, typename, value))
    else:
        print('%s%-*s = %s%s' % (prefix, max_name_length, name, typename, value))