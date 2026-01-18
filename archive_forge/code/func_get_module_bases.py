from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexInput, HexDump
from winappdbg.util import PathOperations
import os
import warnings
import traceback
def get_module_bases(self):
    """
        @see:    L{iter_module_addresses}
        @rtype:  list( int... )
        @return: List of DLL base addresses in this snapshot.
        """
    self.__initialize_snapshot()
    return compat.keys(self.__moduleDict)