from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def open_handle(self, dwDesiredAccess=win32.THREAD_ALL_ACCESS):
    """
        Opens a new handle to the thread, closing the previous one.

        The new handle is stored in the L{hThread} property.

        @warn: Normally you should call L{get_handle} instead, since it's much
            "smarter" and tries to reuse handles and merge access rights.

        @type  dwDesiredAccess: int
        @param dwDesiredAccess: Desired access rights.
            Defaults to L{win32.THREAD_ALL_ACCESS}.
            See: U{http://msdn.microsoft.com/en-us/library/windows/desktop/ms686769(v=vs.85).aspx}

        @raise WindowsError: It's not possible to open a handle to the thread
            with the requested access rights. This tipically happens because
            the target thread belongs to system process and the debugger is not
            runnning with administrative rights.
        """
    hThread = win32.OpenThread(dwDesiredAccess, win32.FALSE, self.dwThreadId)
    if not hasattr(self.hThread, '__del__'):
        self.close_handle()
    self.hThread = hThread