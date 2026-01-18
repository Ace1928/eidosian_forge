from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def is_converting_constructor(self):
    """Returns True if the cursor refers to a C++ converting constructor.
        """
    return conf.lib.clang_CXXConstructor_isConvertingConstructor(self)