from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def is_scoped_enum(self):
    """Returns True if the cursor refers to a scoped enum declaration.
        """
    return conf.lib.clang_EnumDecl_isScoped(self)