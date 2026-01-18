from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
@staticmethod
def from_position(tu, file, line, column):
    """
        Retrieve the source location associated with a given file/line/column in
        a particular translation unit.
        """
    return conf.lib.clang_getLocation(tu, file, line, column)