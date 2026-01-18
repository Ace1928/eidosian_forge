from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def remove_last_match(self, address, size):
    """
        Removes the last buffer from the watch object
        to match the given address and size.

        @type  address: int
        @param address: Memory address of buffer to stop watching.

        @type  size: int
        @param size: Size in bytes of buffer to stop watching.

        @rtype:  int
        @return: Number of matching elements found. Only the last one to be
            added is actually deleted upon calling this method.

            This counter allows you to know if there are more matching elements
            and how many.
        """
    count = 0
    start = address
    end = address + size - 1
    matched = None
    for item in self.__ranges:
        if item.match(start) and item.match(end):
            matched = item
            count += 1
    self.__ranges.remove(matched)
    return count