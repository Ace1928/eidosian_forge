from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
@property
def objc_type_encoding(self):
    """Return the Objective-C type encoding as a str."""
    if not hasattr(self, '_objc_type_encoding'):
        self._objc_type_encoding = conf.lib.clang_getDeclObjCTypeEncoding(self)
    return self._objc_type_encoding