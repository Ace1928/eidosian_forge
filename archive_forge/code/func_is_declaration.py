from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def is_declaration(self):
    """Test if this is a declaration kind."""
    return conf.lib.clang_isDeclaration(self)