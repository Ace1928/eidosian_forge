from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump, HexInput
from winappdbg.util import Regenerator, PathOperations, MemoryAddresses
from winappdbg.module import Module, _ModuleContainer
from winappdbg.thread import Thread, _ThreadContainer
from winappdbg.window import Window
from winappdbg.search import Search, \
from winappdbg.disasm import Disassembler
import re
import os
import os.path
import ctypes
import struct
import warnings
import traceback
def close_process_and_thread_handles(self):
    """
        Closes all open handles to processes and threads in this snapshot.
        """
    for aProcess in self.iter_processes():
        aProcess.close_thread_handles()
        try:
            aProcess.close_handle()
        except Exception:
            e = sys.exc_info()[1]
            try:
                msg = 'Cannot close process handle %s, reason: %s'
                msg %= (aProcess.hProcess.value, str(e))
                warnings.warn(msg)
            except Exception:
                pass