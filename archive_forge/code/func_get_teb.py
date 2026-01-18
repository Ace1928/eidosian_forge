from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def get_teb(self):
    """
        Returns a copy of the TEB.
        To dereference pointers in it call L{Process.read_structure}.

        @rtype:  L{TEB}
        @return: TEB structure.
        @raise WindowsError: An exception is raised on error.
        """
    return self.get_process().read_structure(self.get_teb_address(), win32.TEB)