from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def get_register(self, register):
    """
        @type  register: str
        @param register: Register name.

        @rtype:  int
        @return: Value of the requested register.
        """
    'Returns the value of a specific register.'
    context = self.get_context()
    return context[register]