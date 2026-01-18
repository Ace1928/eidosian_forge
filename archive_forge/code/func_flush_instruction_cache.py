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
def flush_instruction_cache(self):
    """
        Flush the instruction cache. This is required if the process memory is
        modified and one or more threads are executing nearby the modified
        memory region.

        @see: U{http://blogs.msdn.com/oldnewthing/archive/2003/12/08/55954.aspx#55958}

        @raise WindowsError: Raises exception on error.
        """
    win32.FlushInstructionCache(self.get_handle())