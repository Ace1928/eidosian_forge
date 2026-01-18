from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def get_code_breakpoint(self, dwProcessId, address):
    """
        Returns the internally used breakpoint object,
        for the code breakpoint defined at the given address.

        @warning: It's usually best to call the L{Debug} methods
            instead of accessing the breakpoint objects directly.

        @see:
            L{define_code_breakpoint},
            L{has_code_breakpoint},
            L{enable_code_breakpoint},
            L{enable_one_shot_code_breakpoint},
            L{disable_code_breakpoint},
            L{erase_code_breakpoint}

        @type  dwProcessId: int
        @param dwProcessId: Process global ID.

        @type  address: int
        @param address: Memory address where the breakpoint is defined.

        @rtype:  L{CodeBreakpoint}
        @return: The code breakpoint object.
        """
    key = (dwProcessId, address)
    if key not in self.__codeBP:
        msg = 'No breakpoint at process %d, address %s'
        address = HexDump.address(address)
        raise KeyError(msg % (dwProcessId, address))
    return self.__codeBP[key]