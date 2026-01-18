from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def stop_tracing_all(self):
    """
        Stop tracing mode for all threads in all debugees.
        """
    for pid in self.get_debugee_pids():
        self.stop_tracing_process(pid)