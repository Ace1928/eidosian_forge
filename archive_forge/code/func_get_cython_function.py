from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
@default_selected_gdb_frame()
def get_cython_function(self, frame):
    result = self.cy.functions_by_cname.get(frame.name())
    if result is None:
        raise NoCythonFunctionInFrameError()
    return result