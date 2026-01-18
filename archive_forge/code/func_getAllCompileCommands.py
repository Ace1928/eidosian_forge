from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def getAllCompileCommands(self):
    """
        Get an iterable object providing all the CompileCommands available from
        the database.
        """
    return conf.lib.clang_CompilationDatabase_getAllCompileCommands(self)