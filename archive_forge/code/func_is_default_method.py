from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def is_default_method(self):
    """Returns True if the cursor refers to a C++ member function or member
        function template that is declared '= default'.
        """
    return conf.lib.clang_CXXMethod_isDefaulted(self)