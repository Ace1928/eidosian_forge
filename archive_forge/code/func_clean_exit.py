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
def clean_exit(self, dwExitCode=0, bWait=False, dwTimeout=None):
    """
        Injects a new thread to call ExitProcess().
        Optionally waits for the injected thread to finish.

        @warning: Setting C{bWait} to C{True} when the process is frozen by a
            debug event will cause a deadlock in your debugger.

        @type  dwExitCode: int
        @param dwExitCode: Process exit code.

        @type  bWait: bool
        @param bWait: C{True} to wait for the process to finish.
            C{False} to return immediately.

        @type  dwTimeout: int
        @param dwTimeout: (Optional) Timeout value in milliseconds.
            Ignored if C{bWait} is C{False}.

        @raise WindowsError: An exception is raised on error.
        """
    if not dwExitCode:
        dwExitCode = 0
    pExitProcess = self.resolve_label('kernel32!ExitProcess')
    aThread = self.start_thread(pExitProcess, dwExitCode)
    if bWait:
        aThread.wait(dwTimeout)