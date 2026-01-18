from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def peek_code_bytes(self, size=128, offset=0):
    """
        Tries to read some bytes of the code currently being executed.

        @type  size: int
        @param size: Number of bytes to read.

        @type  offset: int
        @param offset: Offset from the program counter to begin reading.

        @rtype:  str
        @return: Bytes read from the process memory.
            May be less than the requested number of bytes.
        """
    return self.get_process().peek(self.get_pc() + offset, size)