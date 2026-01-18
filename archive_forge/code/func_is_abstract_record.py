from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def is_abstract_record(self):
    """Returns True if the cursor refers to a C++ record declaration
        that has pure virtual member functions.
        """
    return conf.lib.clang_CXXRecord_isAbstract(self)