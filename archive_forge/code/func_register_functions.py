from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def register_functions(lib, ignore_errors):
    """Register function prototypes with a libclang library instance.

    This must be called as part of library instantiation so Python knows how
    to call out to the shared library.
    """

    def register(item):
        return register_function(lib, item, ignore_errors)
    for f in functionList:
        register(f)