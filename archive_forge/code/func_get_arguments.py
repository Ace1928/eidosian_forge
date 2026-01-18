from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def get_arguments(self):
    """Return an iterator for accessing the arguments of this cursor."""
    num_args = conf.lib.clang_Cursor_getNumArguments(self)
    for i in range(0, num_args):
        yield conf.lib.clang_Cursor_getArgument(self, i)