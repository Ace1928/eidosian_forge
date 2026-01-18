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
def get_ntstatus_code(self):
    """
        @rtype:  int
        @return: NTSTATUS status code that caused the exception.

        @note: This method is only meaningful for in-page memory error
            exceptions.

        @raise NotImplementedError: Not an in-page memory error.
        """
    if self.get_exception_code() != win32.EXCEPTION_IN_PAGE_ERROR:
        msg = 'This method is only meaningful for in-page memory error exceptions.'
        raise NotImplementedError(msg)
    return self.get_exception_information(2)