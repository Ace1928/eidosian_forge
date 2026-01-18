from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def is_copy_constructor(self):
    """Returns True if the cursor refers to a C++ copy constructor.
        """
    return conf.lib.clang_CXXConstructor_isCopyConstructor(self)