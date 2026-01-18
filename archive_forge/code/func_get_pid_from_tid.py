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
def get_pid_from_tid(self, dwThreadId):
    """
        Retrieves the global ID of the process that owns the thread.

        @type  dwThreadId: int
        @param dwThreadId: Thread global ID.

        @rtype:  int
        @return: Process global ID.

        @raise KeyError: The thread does not exist.
        """
    try:
        try:
            hThread = win32.OpenThread(win32.THREAD_QUERY_LIMITED_INFORMATION, False, dwThreadId)
        except WindowsError:
            e = sys.exc_info()[1]
            if e.winerror != win32.ERROR_ACCESS_DENIED:
                raise
            hThread = win32.OpenThread(win32.THREAD_QUERY_INFORMATION, False, dwThreadId)
        try:
            return win32.GetProcessIdOfThread(hThread)
        finally:
            hThread.close()
    except Exception:
        for aProcess in self.iter_processes():
            if aProcess.has_thread(dwThreadId):
                return aProcess.get_pid()
    self.scan_processes_and_threads()
    for aProcess in self.iter_processes():
        if aProcess.has_thread(dwThreadId):
            return aProcess.get_pid()
    msg = 'Unknown thread ID %d' % dwThreadId
    raise KeyError(msg)