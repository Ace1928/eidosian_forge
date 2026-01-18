from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def set_action(self, action=None):
    """
        Sets a new action callback for the breakpoint.

        @type  action: function
        @param action: (Optional) Action callback function.
        """
    self.__action = action