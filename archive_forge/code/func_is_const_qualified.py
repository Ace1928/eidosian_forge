from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def is_const_qualified(self):
    """Determine whether a Type has the "const" qualifier set.

        This does not look through typedefs that may have added "const"
        at a different level.
        """
    return conf.lib.clang_isConstQualifiedType(self)