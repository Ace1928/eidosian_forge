from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
@property
def mangled_name(self):
    """Return the mangled name for the entity referenced by this cursor."""
    if not hasattr(self, '_mangled_name'):
        self._mangled_name = conf.lib.clang_Cursor_getMangling(self)
    return self._mangled_name