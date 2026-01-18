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
def get_thread_handle(self):
    """
        @rtype:  L{ThreadHandle}
        @return: Thread handle received from the system.
            Returns C{None} if the handle is not available.
        """
    hThread = self.raw.u.CreateProcessInfo.hThread
    if hThread in (0, win32.NULL, win32.INVALID_HANDLE_VALUE):
        hThread = None
    else:
        hThread = ThreadHandle(hThread, False, win32.THREAD_ALL_ACCESS)
    return hThread