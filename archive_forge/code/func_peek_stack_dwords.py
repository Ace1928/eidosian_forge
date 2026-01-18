from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def peek_stack_dwords(self, count, offset=0):
    """
        Tries to read DWORDs from the top of the stack.

        @type  count: int
        @param count: Number of DWORDs to read.

        @type  offset: int
        @param offset: Offset from the stack pointer to begin reading.

        @rtype:  tuple( int... )
        @return: Tuple of integers read from the stack.
            May be less than the requested number of DWORDs.
        """
    stackData = self.peek_stack_data(count * 4, offset)
    if len(stackData) & 3:
        stackData = stackData[:-len(stackData) & 3]
    if not stackData:
        return ()
    return struct.unpack('<' + 'L' * count, stackData)