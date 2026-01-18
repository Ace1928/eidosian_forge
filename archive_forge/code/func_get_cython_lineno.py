from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
@default_selected_gdb_frame()
def get_cython_lineno(self, frame):
    """
        Get the current Cython line number. Returns 0 if there is no
        correspondence between the C and Cython code.
        """
    cyfunc = self.get_cython_function(frame)
    return cyfunc.module.lineno_c2cy.get(self.get_c_lineno(frame), 0)