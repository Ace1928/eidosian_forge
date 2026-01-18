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
def get_exception_information(self, index):
    """
        @type  index: int
        @param index: Index into the exception information block.

        @rtype:  int
        @return: Exception information DWORD.
        """
    if index < 0 or index > win32.EXCEPTION_MAXIMUM_PARAMETERS:
        raise IndexError('Array index out of range: %s' % repr(index))
    info = self.raw.u.Exception.ExceptionRecord.ExceptionInformation
    value = info[index]
    if value is None:
        value = 0
    return value