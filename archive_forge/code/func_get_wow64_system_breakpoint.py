from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexInput, HexDump
from winappdbg.util import PathOperations
import os
import warnings
import traceback
def get_wow64_system_breakpoint(self):
    """
        @rtype:  int or None
        @return: Memory address of the Wow64 system breakpoint
            within the process address space.
            Returns C{None} on error.
        """
    return self.__get_system_breakpoint('ntdll32!DbgBreakPoint')