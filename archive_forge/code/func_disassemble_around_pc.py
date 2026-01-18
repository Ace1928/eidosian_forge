from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def disassemble_around_pc(self, dwSize=64):
    """
        Disassemble around the program counter of the given thread.

        @type  dwSize: int
        @param dwSize: Delta offset.
            Code will be read from pc - dwSize to pc + dwSize.

        @rtype:  list of tuple( long, int, str, str )
        @return: List of tuples. Each tuple represents an assembly instruction
            and contains:
             - Memory address of instruction.
             - Size of instruction in bytes.
             - Disassembly line of instruction.
             - Hexadecimal dump of instruction.
        """
    aProcess = self.get_process()
    return aProcess.disassemble_around(self.get_pc(), dwSize)