from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
@default_selected_gdb_frame(err=False)
def is_python_function(self, frame):
    """
        Tells if a frame is associated with a Python function.
        If we can't read the Python frame information, don't regard it as such.
        """
    if frame.name() == 'PyEval_EvalFrameEx':
        pyframe = libpython.Frame(frame).get_pyop()
        return pyframe and (not pyframe.is_optimized_out())
    return False