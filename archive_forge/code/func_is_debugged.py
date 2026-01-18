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
def is_debugged(self):
    """
        Tries to determine if the process is being debugged by another process.
        It may detect other debuggers besides WinAppDbg.

        @rtype:  bool
        @return: C{True} if the process has a debugger attached.

        @warning:
            May return inaccurate results when some anti-debug techniques are
            used by the target process.

        @note: To know if a process currently being debugged by a L{Debug}
            object, call L{Debug.is_debugee} instead.
        """
    hProcess = self.get_handle(win32.PROCESS_QUERY_INFORMATION)
    return win32.CheckRemoteDebuggerPresent(hProcess)