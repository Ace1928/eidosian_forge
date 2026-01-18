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
def is_optimized_out(self):
    """
        Is the value of the underlying PyObject* visible to the debugger?

        This can vary with the precise version of the compiler used to build
        Python, and the precise version of gdb.

        See e.g. https://bugzilla.redhat.com/show_bug.cgi?id=556975 with
        PyEval_EvalFrameEx's "f"
        """
    return self._gdbval.is_optimized_out