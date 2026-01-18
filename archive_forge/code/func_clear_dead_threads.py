from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def clear_dead_threads(self):
    """
        Remove Thread objects from the snapshot
        referring to threads no longer running.
        """
    for tid in self.get_thread_ids():
        aThread = self.get_thread(tid)
        if not aThread.is_alive():
            self._del_thread(aThread)