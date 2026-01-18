from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def set_fp(self, fp):
    """
            Sets the value of the frame pointer register.

            @type  fp: int
            @param fp: Value of the frame pointer register.
            """
    flags = win32.CONTEXT_CONTROL | win32.CONTEXT_INTEGER
    context = self.get_context(flags)
    context.fp = fp
    self.set_context(context)