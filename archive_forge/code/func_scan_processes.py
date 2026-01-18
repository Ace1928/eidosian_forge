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
def scan_processes(self):
    """
        Populates the snapshot with running processes.

        Tipically you don't need to call this method directly, if unsure use
        L{scan} instead.

        @note: This method uses the Remote Desktop API instead of the Toolhelp
            API. It might give slightly different results, especially if the
            current process does not have full privileges.

        @note: This method will only retrieve process filenames. To get the
            process pathnames instead, B{after} this method call
            L{scan_process_filenames}.

        @raise WindowsError: An error occured while updating the snapshot.
            The snapshot was not modified.
        """
    our_pid = win32.GetCurrentProcessId()
    dead_pids = set(compat.iterkeys(self.__processDict))
    if our_pid in dead_pids:
        dead_pids.remove(our_pid)
    pProcessInfo = None
    try:
        pProcessInfo, dwCount = win32.WTSEnumerateProcesses(win32.WTS_CURRENT_SERVER_HANDLE)
        for index in compat.xrange(dwCount):
            sProcessInfo = pProcessInfo[index]
            pid = sProcessInfo.ProcessId
            if pid == our_pid:
                continue
            if pid in dead_pids:
                dead_pids.remove(pid)
            fileName = sProcessInfo.pProcessName
            if pid not in self.__processDict:
                aProcess = Process(pid, fileName=fileName)
                self._add_process(aProcess)
            elif fileName:
                aProcess = self.__processDict.get(pid)
                if not aProcess.fileName:
                    aProcess.fileName = fileName
    finally:
        if pProcessInfo is not None:
            try:
                win32.WTSFreeMemory(pProcessInfo)
            except WindowsError:
                pass
    for pid in dead_pids:
        self._del_process(pid)