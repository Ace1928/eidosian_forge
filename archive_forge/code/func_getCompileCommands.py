from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def getCompileCommands(self, filename):
    """
        Get an iterable object providing all the CompileCommands available to
        build filename. Returns None if filename is not found in the database.
        """
    return conf.lib.clang_CompilationDatabase_getCompileCommands(self, fspath(filename))