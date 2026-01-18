from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def enable_hardware_breakpoint(self, dwThreadId, address):
    """
        Enables the hardware breakpoint at the given address.

        @see:
            L{define_hardware_breakpoint},
            L{has_hardware_breakpoint},
            L{get_hardware_breakpoint},
            L{enable_one_shot_hardware_breakpoint},
            L{disable_hardware_breakpoint}
            L{erase_hardware_breakpoint},

        @note: Do not set hardware breakpoints while processing the system
            breakpoint event.

        @type  dwThreadId: int
        @param dwThreadId: Thread global ID.

        @type  address: int
        @param address: Memory address of breakpoint.
        """
    t = self.system.get_thread(dwThreadId)
    bp = self.get_hardware_breakpoint(dwThreadId, address)
    if bp.is_running():
        self.__del_running_bp_from_all_threads(bp)
    bp.enable(None, t)