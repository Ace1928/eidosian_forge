from __future__ import print_function
import gdb
import os
import locale
import sys
import sys
import libpython
import re
import warnings
import tempfile
import functools
import textwrap
import itertools
import traceback
def iter_locals(self):
    """
        Yield a sequence of (name,value) pairs of PyObjectPtr instances, for
        the local variables of this frame
        """
    if self.is_optimized_out():
        return
    f_localsplus = self.field('f_localsplus')
    for i in safe_range(self.co_nlocals):
        pyop_value = PyObjectPtr.from_pyobject_ptr(f_localsplus[i])
        if not pyop_value.is_null():
            pyop_name = PyObjectPtr.from_pyobject_ptr(self.co_varnames[i])
            yield (pyop_name, pyop_value)