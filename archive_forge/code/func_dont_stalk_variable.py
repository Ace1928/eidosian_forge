from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def dont_stalk_variable(self, tid, address):
    """
        Clears a hardware breakpoint set by L{stalk_variable}.

        @type  tid: int
        @param tid: Thread global ID.

        @type  address: int
        @param address: Memory address of variable to stop watching.
        """
    self.__clear_variable_watch(tid, address)