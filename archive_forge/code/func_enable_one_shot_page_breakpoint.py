from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def enable_one_shot_page_breakpoint(self, dwProcessId, address):
    """
        Enables the page breakpoint at the given address for only one shot.

        @see:
            L{define_page_breakpoint},
            L{has_page_breakpoint},
            L{get_page_breakpoint},
            L{enable_page_breakpoint},
            L{disable_page_breakpoint}
            L{erase_page_breakpoint},

        @type  dwProcessId: int
        @param dwProcessId: Process global ID.

        @type  address: int
        @param address: Memory address of breakpoint.
        """
    p = self.system.get_process(dwProcessId)
    bp = self.get_page_breakpoint(dwProcessId, address)
    if bp.is_running():
        self.__del_running_bp_from_all_threads(bp)
    bp.one_shot(p, None)