from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def get_condition(self):
    """
        @rtype:  bool, function
        @return: Returns the condition callback for conditional breakpoints.
            Returns C{True} for unconditional breakpoints.
        """
    return self.__condition