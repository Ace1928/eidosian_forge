from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
@staticmethod
def from_offset(tu, file, offset):
    """Retrieve a SourceLocation from a given character offset.

        tu -- TranslationUnit file belongs to
        file -- File instance to obtain offset from
        offset -- Integer character offset within file
        """
    return conf.lib.clang_getLocationForOffset(tu, file, offset)