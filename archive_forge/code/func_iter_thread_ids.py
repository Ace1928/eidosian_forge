from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def iter_thread_ids(self):
    """
        @see:    L{iter_threads}
        @rtype:  dictionary-keyiterator
        @return: Iterator of global thread IDs in this snapshot.
        """
    self.__initialize_snapshot()
    return compat.iterkeys(self.__threadDict)