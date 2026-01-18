from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def is_pod(self):
    """Determine whether this Type represents plain old data (POD)."""
    return conf.lib.clang_isPODType(self)