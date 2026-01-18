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
def scan_process_filenames(self):
    """
        Update the filename for each process in the snapshot when possible.

        @note: Tipically you don't need to call this method. It's called
            automatically by L{scan} to get the full pathname for each process
            when possible, since some scan methods only get filenames without
            the path component.

            If unsure, use L{scan} instead.

        @see: L{scan}, L{Process.get_filename}

        @rtype: bool
        @return: C{True} if all the pathnames were retrieved, C{False} if the
            debugger doesn't have permission to scan some processes. In either
            case, all processes the debugger has access to have a full pathname
            instead of just a filename.
        """
    complete = True
    for aProcess in self.__processDict.values():
        try:
            new_name = None
            old_name = aProcess.fileName
            try:
                aProcess.fileName = None
                new_name = aProcess.get_filename()
            finally:
                if not new_name:
                    aProcess.fileName = old_name
                    complete = False
        except Exception:
            complete = False
    return complete