from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump, HexInput
from winappdbg.util import Regenerator, PathOperations, MemoryAddresses
from winappdbg.module import Module, _ModuleContainer
from winappdbg.thread import Thread, _ThreadContainer
from winappdbg.window import Window
from winappdbg.search import Search, \
from winappdbg.disasm import Disassembler
import re
import os
import os.path
import ctypes
import struct
import warnings
import traceback
def get_environment_data(self, fUnicode=None):
    """
        Retrieves the environment block data with wich the program is running.

        @warn: Deprecated since WinAppDbg 1.5.

        @see: L{win32.GuessStringType}

        @type  fUnicode: bool or None
        @param fUnicode: C{True} to return a list of Unicode strings, C{False}
            to return a list of ANSI strings, or C{None} to return whatever
            the default is for string types.

        @rtype:  list of str
        @return: Environment keys and values separated by a (C{=}) character,
            as found in the process memory.

        @raise WindowsError: On error an exception is raised.
        """
    warnings.warn('Process.get_environment_data() is deprecated since WinAppDbg 1.5.', DeprecationWarning)
    block = [key + u'=' + value for key, value in self.get_environment_variables()]
    if fUnicode is None:
        gst = win32.GuessStringType
        fUnicode = gst.t_default == gst.t_unicode
    if not fUnicode:
        block = [x.encode('cp1252') for x in block]
    return block