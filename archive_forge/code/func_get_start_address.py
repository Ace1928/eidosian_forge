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
def get_start_address(self):
    """
        @rtype:  int
        @return: Pointer to the first instruction to execute in this process.

            Returns C{NULL} when the debugger attaches to a process.

            See U{http://msdn.microsoft.com/en-us/library/ms679295(VS.85).aspx}
        """
    return self.raw.u.CreateProcessInfo.lpStartAddress