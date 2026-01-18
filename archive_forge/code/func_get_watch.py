from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def get_watch(self):
    """
        @see: L{validWatchSizes}
        @rtype:  int
        @return: The breakpoint watch flag.
        """
    return self.__watch