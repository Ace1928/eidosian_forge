from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def get_all_code_breakpoints(self):
    """
        @rtype:  list of tuple( int, L{CodeBreakpoint} )
        @return: All code breakpoints as a list of tuples (pid, bp).
        """
    return [(pid, bp) for (pid, address), bp in compat.iteritems(self.__codeBP)]