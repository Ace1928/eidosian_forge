from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def get_page_breakpoint(self, dwProcessId, address):
    """
        Returns the internally used breakpoint object,
        for the page breakpoint defined at the given address.

        @warning: It's usually best to call the L{Debug} methods
            instead of accessing the breakpoint objects directly.

        @see:
            L{define_page_breakpoint},
            L{has_page_breakpoint},
            L{enable_page_breakpoint},
            L{enable_one_shot_page_breakpoint},
            L{disable_page_breakpoint},
            L{erase_page_breakpoint}

        @type  dwProcessId: int
        @param dwProcessId: Process global ID.

        @type  address: int
        @param address: Memory address where the breakpoint is defined.

        @rtype:  L{PageBreakpoint}
        @return: The page breakpoint object.
        """
    key = (dwProcessId, address)
    if key not in self.__pageBP:
        msg = 'No breakpoint at process %d, address %s'
        address = HexDump.addresS(address)
        raise KeyError(msg % (dwProcessId, address))
    return self.__pageBP[key]