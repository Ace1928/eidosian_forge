from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
@default_selected_gdb_frame()
def get_c_lineno(self, frame):
    return frame.find_sal().line