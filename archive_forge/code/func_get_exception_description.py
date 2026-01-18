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
def get_exception_description(self):
    """
        @rtype:  str
        @return: User-friendly name of the exception.
        """
    code = self.get_exception_code()
    description = self.__exceptionDescription.get(code, None)
    if description is None:
        try:
            description = 'Exception code %s (%s)'
            description = description % (HexDump.integer(code), ctypes.FormatError(code))
        except OverflowError:
            description = 'Exception code %s' % HexDump.integer(code)
    return description