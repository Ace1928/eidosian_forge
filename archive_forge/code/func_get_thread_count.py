from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def get_thread_count(self):
    """
        @rtype:  int
        @return: Count of L{Thread} objects in this snapshot.
        """
    self.__initialize_snapshot()
    return len(self.__threadDict)