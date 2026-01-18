from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def stop_tracing(self, tid):
    """
        Stop tracing mode in the given thread.

        @type  tid: int
        @param tid: Global ID of thread to stop tracing.
        """
    if self.is_tracing(tid):
        thread = self.system.get_thread(tid)
        self.__stop_tracing(thread)