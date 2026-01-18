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
def scan_processes_fast(self):
    """
        Populates the snapshot with running processes.
        Only the PID is retrieved for each process.

        Dead processes are removed.
        Threads and modules of living processes are ignored.

        Tipically you don't need to call this method directly, if unsure use
        L{scan} instead.

        @note: This method uses the PSAPI. It may be faster for scanning,
            but some information may be missing, outdated or slower to obtain.
            This could be a good tradeoff under some circumstances.
        """
    new_pids = set(win32.EnumProcesses())
    old_pids = set(compat.iterkeys(self.__processDict))
    our_pid = win32.GetCurrentProcessId()
    if our_pid in new_pids:
        new_pids.remove(our_pid)
    if our_pid in old_pids:
        old_pids.remove(our_pid)
    for pid in new_pids.difference(old_pids):
        self._add_process(Process(pid))
    for pid in old_pids.difference(new_pids):
        self._del_process(pid)