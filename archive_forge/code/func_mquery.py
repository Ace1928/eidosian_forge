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
def mquery(self, lpAddress):
    """
        Query memory information from the address space of the process.
        Returns a L{win32.MemoryBasicInformation} object.

        @see: U{http://msdn.microsoft.com/en-us/library/aa366907(VS.85).aspx}

        @type  lpAddress: int
        @param lpAddress: Address of memory to query.

        @rtype:  L{win32.MemoryBasicInformation}
        @return: Memory region information.

        @raise WindowsError: On error an exception is raised.
        """
    hProcess = self.get_handle(win32.PROCESS_QUERY_INFORMATION)
    return win32.VirtualQueryEx(hProcess, lpAddress)