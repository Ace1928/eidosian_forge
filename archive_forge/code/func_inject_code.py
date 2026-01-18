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
def inject_code(self, payload, lpParameter=0):
    """
        Injects relocatable code into the process memory and executes it.

        @warning: Don't forget to free the memory when you're done with it!
            Otherwise you'll be leaking memory in the target process.

        @see: L{inject_dll}

        @type  payload: str
        @param payload: Relocatable code to run in a new thread.

        @type  lpParameter: int
        @param lpParameter: (Optional) Parameter to be pushed in the stack.

        @rtype:  tuple( L{Thread}, int )
        @return: The injected Thread object
            and the memory address where the code was written.

        @raise WindowsError: An exception is raised on error.
        """
    lpStartAddress = self.malloc(len(payload))
    try:
        self.write(lpStartAddress, payload)
        aThread = self.start_thread(lpStartAddress, lpParameter, bSuspended=False)
        aThread.pInjectedMemory = lpStartAddress
    except Exception:
        self.free(lpStartAddress)
        raise
    return (aThread, lpStartAddress)