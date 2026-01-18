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
def has_process(self, dwProcessId):
    """
        @type  dwProcessId: int
        @param dwProcessId: Global ID of the process to look for.

        @rtype:  bool
        @return: C{True} if the snapshot contains a
            L{Process} object with the given global ID.
        """
    self.__initialize_snapshot()
    return dwProcessId in self.__processDict