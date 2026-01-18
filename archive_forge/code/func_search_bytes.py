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
def search_bytes(self, bytes, minAddr=None, maxAddr=None):
    """
        Search for the given byte pattern within the process memory.

        @type  bytes: str
        @param bytes: Bytes to search for.

        @type  minAddr: int
        @param minAddr: (Optional) Start the search at this memory address.

        @type  maxAddr: int
        @param maxAddr: (Optional) Stop the search at this memory address.

        @rtype:  iterator of int
        @return: An iterator of memory addresses where the pattern was found.

        @raise WindowsError: An error occurred when querying or reading the
            process memory.
        """
    pattern = BytePattern(bytes)
    matches = Search.search_process(self, pattern, minAddr, maxAddr)
    for addr, size, data in matches:
        yield addr