from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def get_all_breakpoints(self):
    """
        Returns all breakpoint objects as a list of tuples.

        Each tuple contains:
         - Process global ID to which the breakpoint applies.
         - Thread global ID to which the breakpoint applies, or C{None}.
         - The L{Breakpoint} object itself.

        @note: If you're only interested in a specific breakpoint type, or in
            breakpoints for a specific process or thread, it's probably faster
            to call one of the following methods:
             - L{get_all_code_breakpoints}
             - L{get_all_page_breakpoints}
             - L{get_all_hardware_breakpoints}
             - L{get_process_code_breakpoints}
             - L{get_process_page_breakpoints}
             - L{get_process_hardware_breakpoints}
             - L{get_thread_hardware_breakpoints}

        @rtype:  list of tuple( pid, tid, bp )
        @return: List of all breakpoints.
        """
    bplist = list()
    for pid, bp in self.get_all_code_breakpoints():
        bplist.append((pid, None, bp))
    for pid, bp in self.get_all_page_breakpoints():
        bplist.append((pid, None, bp))
    for tid, bp in self.get_all_hardware_breakpoints():
        pid = self.system.get_thread(tid).get_pid()
        bplist.append((pid, tid, bp))
    return bplist