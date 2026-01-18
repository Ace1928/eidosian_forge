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
def iter_memory_map(self, minAddr=None, maxAddr=None):
    """
        Produces an iterator over the memory map to the process address space.

        Optionally restrict the map to the given address range.

        @see: L{mquery}

        @type  minAddr: int
        @param minAddr: (Optional) Starting address in address range to query.

        @type  maxAddr: int
        @param maxAddr: (Optional) Ending address in address range to query.

        @rtype:  iterator of L{win32.MemoryBasicInformation}
        @return: List of memory region information objects.
        """
    minAddr, maxAddr = MemoryAddresses.align_address_range(minAddr, maxAddr)
    prevAddr = minAddr - 1
    currentAddr = minAddr
    while prevAddr < currentAddr < maxAddr:
        try:
            mbi = self.mquery(currentAddr)
        except WindowsError:
            e = sys.exc_info()[1]
            if e.winerror == win32.ERROR_INVALID_PARAMETER:
                break
            raise
        yield mbi
        prevAddr = currentAddr
        currentAddr = mbi.BaseAddress + mbi.RegionSize