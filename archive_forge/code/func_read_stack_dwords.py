from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def read_stack_dwords(self, count, offset=0):
    """
        Reads DWORDs from the top of the stack.

        @type  count: int
        @param count: Number of DWORDs to read.

        @type  offset: int
        @param offset: Offset from the stack pointer to begin reading.

        @rtype:  tuple( int... )
        @return: Tuple of integers read from the stack.

        @raise WindowsError: Could not read the requested data.
        """
    if count > 0:
        stackData = self.read_stack_data(count * 4, offset)
        return struct.unpack('<' + 'L' * count, stackData)
    return ()