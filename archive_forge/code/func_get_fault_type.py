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
def get_fault_type(self):
    """
        @rtype:  int
        @return: Access violation type.
            Should be one of the following constants:

             - L{win32.EXCEPTION_READ_FAULT}
             - L{win32.EXCEPTION_WRITE_FAULT}
             - L{win32.EXCEPTION_EXECUTE_FAULT}

        @note: This method is only meaningful for access violation exceptions,
            in-page memory error exceptions and guard page exceptions.

        @raise NotImplementedError: Wrong kind of exception.
        """
    if self.get_exception_code() not in (win32.EXCEPTION_ACCESS_VIOLATION, win32.EXCEPTION_IN_PAGE_ERROR, win32.EXCEPTION_GUARD_PAGE):
        msg = 'This method is not meaningful for %s.'
        raise NotImplementedError(msg % self.get_exception_name())
    return self.get_exception_information(0)