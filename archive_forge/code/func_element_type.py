from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
@property
def element_type(self):
    """Retrieve the Type of elements within this Type.

        If accessed on a type that is not an array, complex, or vector type, an
        exception will be raised.
        """
    result = conf.lib.clang_getElementType(self)
    if result.kind == TypeKind.INVALID:
        raise Exception('Element type not available on this type.')
    return result