from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexInput, HexDump
from winappdbg.util import PathOperations
import os
import warnings
import traceback
def has_module(self, lpBaseOfDll):
    """
        @type  lpBaseOfDll: int
        @param lpBaseOfDll: Base address of the DLL to look for.

        @rtype:  bool
        @return: C{True} if the snapshot contains a
            L{Module} object with the given base address.
        """
    self.__initialize_snapshot()
    return lpBaseOfDll in self.__moduleDict