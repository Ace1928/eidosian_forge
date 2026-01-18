from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexInput, HexDump
from winappdbg.util import PathOperations
import os
import warnings
import traceback
def get_break_on_error_ptr(self):
    """
        @rtype: int
        @return:
            If present, returns the address of the C{g_dwLastErrorToBreakOn}
            global variable for this process. If not, returns C{None}.
        """
    address = self.__get_system_breakpoint('ntdll!g_dwLastErrorToBreakOn')
    if not address:
        address = self.__get_system_breakpoint('kernel32!g_dwLastErrorToBreakOn')
        self.__system_breakpoints['ntdll!g_dwLastErrorToBreakOn'] = address
    return address