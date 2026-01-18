from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def is_invalid(self):
    """Test if this is an invalid kind."""
    return conf.lib.clang_isInvalid(self)