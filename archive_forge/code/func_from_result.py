from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
@staticmethod
def from_result(res, fn, args):
    if not res:
        raise CompilationDatabaseError(0, 'CompilationDatabase loading failed')
    return CompilationDatabase(res)