from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def get_exit_code(self):
    """
        @rtype:  int
        @return: Thread exit code, or C{STILL_ACTIVE} if it's still alive.
        """
    if win32.THREAD_ALL_ACCESS == win32.THREAD_ALL_ACCESS_VISTA:
        dwAccess = win32.THREAD_QUERY_LIMITED_INFORMATION
    else:
        dwAccess = win32.THREAD_QUERY_INFORMATION
    return win32.GetExitCodeThread(self.get_handle(dwAccess))