from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def one_shot(self, aProcess, aThread):
    if not self.is_enabled() and (not self.is_one_shot()):
        self.__set_bp(aThread)
    super(HardwareBreakpoint, self).one_shot(aProcess, aThread)