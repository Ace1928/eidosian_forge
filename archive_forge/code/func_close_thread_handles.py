from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def close_thread_handles(self):
    """
        Closes all open handles to threads in the snapshot.
        """
    for aThread in self.iter_threads():
        try:
            aThread.close_handle()
        except Exception:
            try:
                e = sys.exc_info()[1]
                msg = 'Cannot close thread handle %s, reason: %s'
                msg %= (aThread.hThread.value, str(e))
                warnings.warn(msg)
            except Exception:
                pass