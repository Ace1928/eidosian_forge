from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def get_span(self):
    """
        @rtype:  tuple( int, int )
        @return:
            Starting and ending address of the memory range
            covered by the breakpoint.
        """
    address = self.get_address()
    size = self.get_size()
    return (address, address + size)