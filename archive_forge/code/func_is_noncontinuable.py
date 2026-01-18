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
def is_noncontinuable(self):
    """
        @see: U{http://msdn.microsoft.com/en-us/library/aa363082(VS.85).aspx}

        @rtype:  bool
        @return: C{True} if the exception is noncontinuable,
            C{False} otherwise.

            Attempting to continue a noncontinuable exception results in an
            EXCEPTION_NONCONTINUABLE_EXCEPTION exception to be raised.
        """
    return bool(self.raw.u.Exception.ExceptionRecord.ExceptionFlags & win32.EXCEPTION_NONCONTINUABLE)