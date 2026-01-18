from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
@property
def enum_type(self):
    """Return the integer type of an enum declaration.

        Returns a Type corresponding to an integer. If the cursor is not for an
        enum, this raises.
        """
    if not hasattr(self, '_enum_type'):
        assert self.kind == CursorKind.ENUM_DECL
        self._enum_type = conf.lib.clang_getEnumDeclIntegerType(self)
    return self._enum_type