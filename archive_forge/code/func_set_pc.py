from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def set_pc(self, pc):
    """
            Sets the value of the program counter register.

            @type  pc: int
            @param pc: Value of the program counter register.
            """
    context = self.get_context(win32.CONTEXT_CONTROL)
    context.pc = pc
    self.set_context(context)