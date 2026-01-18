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
def get_start_time(self):
    """
        Determines when has this process started running.

        @rtype:  win32.SYSTEMTIME
        @return: Process start time.
        """
    if win32.PROCESS_ALL_ACCESS == win32.PROCESS_ALL_ACCESS_VISTA:
        dwAccess = win32.PROCESS_QUERY_LIMITED_INFORMATION
    else:
        dwAccess = win32.PROCESS_QUERY_INFORMATION
    hProcess = self.get_handle(dwAccess)
    CreationTime = win32.GetProcessTimes(hProcess)[0]
    return win32.FileTimeToSystemTime(CreationTime)