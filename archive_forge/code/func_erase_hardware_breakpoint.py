from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def erase_hardware_breakpoint(self, dwThreadId, address):
    """
        Erases the hardware breakpoint at the given address.

        @see:
            L{define_hardware_breakpoint},
            L{has_hardware_breakpoint},
            L{get_hardware_breakpoint},
            L{enable_hardware_breakpoint},
            L{enable_one_shot_hardware_breakpoint},
            L{disable_hardware_breakpoint}

        @type  dwThreadId: int
        @param dwThreadId: Thread global ID.

        @type  address: int
        @param address: Memory address of breakpoint.
        """
    bp = self.get_hardware_breakpoint(dwThreadId, address)
    if not bp.is_disabled():
        self.disable_hardware_breakpoint(dwThreadId, address)
    bpSet = self.__hardwareBP[dwThreadId]
    bpSet.remove(bp)
    if not bpSet:
        del self.__hardwareBP[dwThreadId]