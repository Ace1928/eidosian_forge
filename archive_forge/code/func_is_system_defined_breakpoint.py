from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexInput, HexDump
from winappdbg.util import PathOperations
import os
import warnings
import traceback
def is_system_defined_breakpoint(self, address):
    """
        @type  address: int
        @param address: Memory address.

        @rtype:  bool
        @return: C{True} if the given address points to a system defined
            breakpoint. System defined breakpoints are hardcoded into
            system libraries.
        """
    if address:
        module = self.get_module_at_address(address)
        if module:
            return module.match_name('ntdll') or module.match_name('kernel32')
    return False