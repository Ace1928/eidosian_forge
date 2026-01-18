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
def iter_builtins(self):
    """
        Yield a sequence of (name,value) pairs of PyObjectPtr instances, for
        the builtin variables
        """
    if self.is_optimized_out():
        return ()
    pyop_builtins = self.pyop_field('f_builtins')
    return pyop_builtins.iteritems()