from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
@staticmethod
def get_all_kinds():
    """Return all CursorKind enumeration instances."""
    return [x for x in CursorKind._kinds if not x is None]