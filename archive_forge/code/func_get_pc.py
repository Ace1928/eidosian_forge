from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def get_pc(self):
    """
            @rtype:  int
            @return: Value of the program counter register.
            """
    context = self.get_context(win32.CONTEXT_CONTROL)
    return context.pc