from winappdbg import win32
from winappdbg import compat
from winappdbg.win32 import FileHandle, ProcessHandle, ThreadHandle
from winappdbg.breakpoint import ApiHook
from winappdbg.module import Module
from winappdbg.thread import Thread
from winappdbg.process import Process
from winappdbg.textio import HexDump
from winappdbg.util import StaticClass, PathOperations
import sys
import ctypes
import warnings
import traceback
def get_process_handle(self):
    """
        @rtype:  L{ProcessHandle}
        @return: Process handle received from the system.
            Returns C{None} if the handle is not available.
        """
    hProcess = self.raw.u.CreateProcessInfo.hProcess
    if hProcess in (0, win32.NULL, win32.INVALID_HANDLE_VALUE):
        hProcess = None
    else:
        hProcess = ProcessHandle(hProcess, False, win32.PROCESS_ALL_ACCESS)
    return hProcess