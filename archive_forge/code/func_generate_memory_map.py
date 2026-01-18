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
def generate_memory_map(self, minAddr=None, maxAddr=None):
    """
        Returns a L{Regenerator} that can iterate indefinitely over the memory
        map to the process address space.

        Optionally restrict the map to the given address range.

        @see: L{mquery}

        @type  minAddr: int
        @param minAddr: (Optional) Starting address in address range to query.

        @type  maxAddr: int
        @param maxAddr: (Optional) Ending address in address range to query.

        @rtype:  L{Regenerator} of L{win32.MemoryBasicInformation}
        @return: List of memory region information objects.
        """
    return Regenerator(self.iter_memory_map, minAddr, maxAddr)