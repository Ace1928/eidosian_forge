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
def search_regexp(self, regexp, flags=0, minAddr=None, maxAddr=None, bufferPages=-1):
    """
        Search for the given regular expression within the process memory.

        @type  regexp: str
        @param regexp: Regular expression string.

        @type  flags: int
        @param flags: Regular expression flags.

        @type  minAddr: int
        @param minAddr: (Optional) Start the search at this memory address.

        @type  maxAddr: int
        @param maxAddr: (Optional) Stop the search at this memory address.

        @type  bufferPages: int
        @param bufferPages: (Optional) Number of memory pages to buffer when
            performing the search. Valid values are:
             - C{0} or C{None}:
               Automatically determine the required buffer size. May not give
               complete results for regular expressions that match variable
               sized strings.
             - C{> 0}: Set the buffer size, in memory pages.
             - C{< 0}: Disable buffering entirely. This may give you a little
               speed gain at the cost of an increased memory usage. If the
               target process has very large contiguous memory regions it may
               actually be slower or even fail. It's also the only way to
               guarantee complete results for regular expressions that match
               variable sized strings.

        @rtype:  iterator of tuple( int, int, str )
        @return: An iterator of tuples. Each tuple contains the following:
             - The memory address where the pattern was found.
             - The size of the data that matches the pattern.
             - The data that matches the pattern.

        @raise WindowsError: An error occurred when querying or reading the
            process memory.
        """
    pattern = RegExpPattern(regexp, flags)
    return Search.search_process(self, pattern, minAddr, maxAddr, bufferPages)