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
def is_user_defined_exception(self):
    """
        Determines if this is an user-defined exception. User-defined
        exceptions may contain any exception code that is not system reserved.

        Often the exception code is also a valid Win32 error code, but that's
        up to the debugged application.

        @rtype:  bool
        @return: C{True} if the exception is user-defined, C{False} otherwise.
        """
    return self.get_exception_code() & 268435456 == 0