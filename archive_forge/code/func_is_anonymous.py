from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def is_anonymous(self):
    """
        Check if the record is anonymous.
        """
    if self.kind == CursorKind.FIELD_DECL:
        return self.type.get_declaration().is_anonymous()
    return conf.lib.clang_Cursor_isAnonymous(self)