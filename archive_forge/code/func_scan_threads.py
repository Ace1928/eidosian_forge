from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def scan_threads(self):
    """
        Populates the snapshot with running threads.
        """
    dwProcessId = self.get_pid()
    if dwProcessId in (0, 4, 8):
        return
    dead_tids = self._get_thread_ids()
    dwProcessId = self.get_pid()
    hSnapshot = win32.CreateToolhelp32Snapshot(win32.TH32CS_SNAPTHREAD, dwProcessId)
    try:
        te = win32.Thread32First(hSnapshot)
        while te is not None:
            if te.th32OwnerProcessID == dwProcessId:
                dwThreadId = te.th32ThreadID
                if dwThreadId in dead_tids:
                    dead_tids.remove(dwThreadId)
                if not self._has_thread_id(dwThreadId):
                    aThread = Thread(dwThreadId, process=self)
                    self._add_thread(aThread)
            te = win32.Thread32Next(hSnapshot)
    finally:
        win32.CloseHandle(hSnapshot)
    for tid in dead_tids:
        self._del_thread(tid)