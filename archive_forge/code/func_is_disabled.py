from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def is_disabled(self):
    """
        @rtype:  bool
        @return: C{True} if the breakpoint is in L{DISABLED} state.
        """
    return self.get_state() == self.DISABLED