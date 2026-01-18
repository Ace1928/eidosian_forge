from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def get_process_hardware_breakpoints(self, dwProcessId):
    """
        @see: L{get_thread_hardware_breakpoints}

        @type  dwProcessId: int
        @param dwProcessId: Process global ID.

        @rtype:  list of tuple( int, L{HardwareBreakpoint} )
        @return: All hardware breakpoints for each thread in the given process
            as a list of tuples (tid, bp).
        """
    result = list()
    aProcess = self.system.get_process(dwProcessId)
    for dwThreadId in aProcess.iter_thread_ids():
        if dwThreadId in self.__hardwareBP:
            bplist = self.__hardwareBP[dwThreadId]
            for bp in bplist:
                result.append((dwThreadId, bp))
    return result