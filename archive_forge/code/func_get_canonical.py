from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def get_canonical(self):
    """
        Return the canonical type for a Type.

        Clang's type system explicitly models typedefs and all the
        ways a specific type can be represented.  The canonical type
        is the underlying type with all the "sugar" removed.  For
        example, if 'T' is a typedef for 'int', the canonical type for
        'T' would be 'int'.
        """
    return conf.lib.clang_getCanonicalType(self)