from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def get_process_page_breakpoints(self, dwProcessId):
    """
        @type  dwProcessId: int
        @param dwProcessId: Process global ID.

        @rtype:  list of L{PageBreakpoint}
        @return: All page breakpoints for the given process.
        """
    return [bp for (pid, address), bp in compat.iteritems(self.__pageBP) if pid == dwProcessId]