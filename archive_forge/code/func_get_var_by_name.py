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
def get_var_by_name(self, name):
    """
        Look for the named local variable, returning a (PyObjectPtr, scope) pair
        where scope is a string 'local', 'global', 'builtin'

        If not found, return (None, None)
        """
    for pyop_name, pyop_value in self.iter_locals():
        if name == pyop_name.proxyval(set()):
            return (pyop_value, 'local')
    for pyop_name, pyop_value in self.iter_globals():
        if name == pyop_name.proxyval(set()):
            return (pyop_value, 'global')
    for pyop_name, pyop_value in self.iter_builtins():
        if name == pyop_name.proxyval(set()):
            return (pyop_value, 'builtin')
    return (None, None)