from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
@default_selected_gdb_frame(err=False)
def is_cython_function(self, frame):
    return frame.name() in self.cy.functions_by_cname