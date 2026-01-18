from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def get_array_size(self):
    """
        Retrieve the size of the constant array.
        """
    return conf.lib.clang_getArraySize(self)