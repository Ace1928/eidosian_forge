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
def get_mapped_filenames(self, memoryMap=None):
    """
        Retrieves the filenames for memory mapped files in the debugee.

        @type  memoryMap: list( L{win32.MemoryBasicInformation} )
        @param memoryMap: (Optional) Memory map returned by L{get_memory_map}.
            If not given, the current memory map is used.

        @rtype:  dict( int S{->} str )
        @return: Dictionary mapping memory addresses to file names.
            Native filenames are converted to Win32 filenames when possible.
        """
    hProcess = self.get_handle(win32.PROCESS_VM_READ | win32.PROCESS_QUERY_INFORMATION)
    if not memoryMap:
        memoryMap = self.get_memory_map()
    mappedFilenames = dict()
    for mbi in memoryMap:
        if mbi.Type not in (win32.MEM_IMAGE, win32.MEM_MAPPED):
            continue
        baseAddress = mbi.BaseAddress
        fileName = ''
        try:
            fileName = win32.GetMappedFileName(hProcess, baseAddress)
            fileName = PathOperations.native_to_win32_pathname(fileName)
        except WindowsError:
            pass
        mappedFilenames[baseAddress] = fileName
    return mappedFilenames