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
def read_structure(self, lpBaseAddress, stype):
    """
        Reads a ctypes structure from the memory of the process.

        @see: L{read}

        @type  lpBaseAddress: int
        @param lpBaseAddress: Memory address to begin reading.

        @type  stype: class ctypes.Structure or a subclass.
        @param stype: Structure definition.

        @rtype:  int
        @return: Structure instance filled in with data
            read from the process memory.

        @raise WindowsError: On error an exception is raised.
        """
    if type(lpBaseAddress) not in (type(0), type(long(0))):
        lpBaseAddress = ctypes.cast(lpBaseAddress, ctypes.c_void_p)
    data = self.read(lpBaseAddress, ctypes.sizeof(stype))
    buff = ctypes.create_string_buffer(data)
    ptr = ctypes.cast(ctypes.pointer(buff), ctypes.POINTER(stype))
    return ptr.contents