from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def is_pure_virtual_method(self):
    """Returns True if the cursor refers to a C++ member function or member
        function template that is declared pure virtual.
        """
    return conf.lib.clang_CXXMethod_isPureVirtual(self)