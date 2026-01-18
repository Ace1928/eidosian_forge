from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def is_tracing(self, tid):
    """
        @type  tid: int
        @param tid: Thread global ID.

        @rtype:  bool
        @return: C{True} if the thread is being traced, C{False} otherwise.
        """
    return tid in self.__tracing