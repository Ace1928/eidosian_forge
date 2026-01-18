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
def get_running_time(self):
    """
        Determines how long has this process been running.

        @rtype:  long
        @return: Process running time in milliseconds.
        """
    if win32.PROCESS_ALL_ACCESS == win32.PROCESS_ALL_ACCESS_VISTA:
        dwAccess = win32.PROCESS_QUERY_LIMITED_INFORMATION
    else:
        dwAccess = win32.PROCESS_QUERY_INFORMATION
    hProcess = self.get_handle(dwAccess)
    CreationTime, ExitTime, _, _ = win32.GetProcessTimes(hProcess)
    if self.is_alive():
        ExitTime = win32.GetSystemTimeAsFileTime()
    CreationTime = CreationTime.dwLowDateTime + (CreationTime.dwHighDateTime << 32)
    ExitTime = ExitTime.dwLowDateTime + (ExitTime.dwHighDateTime << 32)
    RunningTime = ExitTime - CreationTime
    return RunningTime / 10000