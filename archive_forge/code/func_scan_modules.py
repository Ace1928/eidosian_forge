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
def scan_modules(self):
    """
        Populates the snapshot with loaded modules.

        Tipically you don't need to call this method directly, if unsure use
        L{scan} instead.

        @note: This method uses the Toolhelp API.

        @see: L{scan_processes_and_threads}

        @rtype: bool
        @return: C{True} if the snapshot is complete, C{False} if the debugger
            doesn't have permission to scan some processes. In either case, the
            snapshot is complete for all processes the debugger has access to.
        """
    complete = True
    for aProcess in compat.itervalues(self.__processDict):
        try:
            aProcess.scan_modules()
        except WindowsError:
            complete = False
    return complete