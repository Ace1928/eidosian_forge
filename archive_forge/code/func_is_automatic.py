from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def is_automatic(self):
    """
        @rtype:  bool
        @return: C{True} if the breakpoint has an action callback defined.
        """
    return self.__action is not None