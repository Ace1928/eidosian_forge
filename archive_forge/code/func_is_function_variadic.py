from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def is_function_variadic(self):
    """Determine whether this function Type is a variadic function type."""
    assert self.kind == TypeKind.FUNCTIONPROTO
    return conf.lib.clang_isFunctionTypeVariadic(self)