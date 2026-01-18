from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def is_default_constructor(self):
    """Returns True if the cursor refers to a C++ default constructor.
        """
    return conf.lib.clang_CXXConstructor_isDefaultConstructor(self)