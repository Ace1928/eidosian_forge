from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def stop_tracing_process(self, pid):
    """
        Stop tracing mode for all threads in the given process.

        @type  pid: int
        @param pid: Global ID of process to stop tracing.
        """
    for thread in self.system.get_process(pid).iter_threads():
        self.__stop_tracing(thread)