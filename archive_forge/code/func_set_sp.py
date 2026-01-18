from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def set_sp(self, sp):
    """
            Sets the value of the stack pointer register.

            @type  sp: int
            @param sp: Value of the stack pointer register.
            """
    context = self.get_context(win32.CONTEXT_CONTROL)
    context.sp = sp
    self.set_context(context)