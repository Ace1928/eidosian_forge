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
def is_address_copy_on_write(self, address):
    """
        Determines if an address belongs to a commited, copy-on-write page.
        The page may or may not have additional permissions.

        @note: Returns always C{False} for kernel mode addresses.

        @type  address: int
        @param address: Memory address to query.

        @rtype:  bool
        @return:
            C{True} if the address belongs to a commited, copy-on-write page.

        @raise WindowsError: An exception is raised on error.
        """
    try:
        mbi = self.mquery(address)
    except WindowsError:
        e = sys.exc_info()[1]
        if e.winerror == win32.ERROR_INVALID_PARAMETER:
            return False
        raise
    return mbi.is_copy_on_write()