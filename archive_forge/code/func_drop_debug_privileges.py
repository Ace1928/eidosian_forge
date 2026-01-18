from __future__ import with_statement
from winappdbg import win32
from winappdbg.registry import Registry
from winappdbg.textio import HexInput, HexDump
from winappdbg.util import Regenerator, PathOperations, MemoryAddresses, DebugRegister, \
from winappdbg.process import _ProcessContainer
from winappdbg.window import Window
import sys
import os
import ctypes
import warnings
from os import path, getenv
@classmethod
def drop_debug_privileges(cls, bIgnoreExceptions=False):
    """
        Drops debug privileges.

        This may be needed to avoid being detected
        by certain anti-debug tricks.

        @type  bIgnoreExceptions: bool
        @param bIgnoreExceptions: C{True} to ignore any exceptions that may be
            raised when dropping debug privileges.

        @rtype:  bool
        @return: C{True} on success, C{False} on failure.

        @raise WindowsError: Raises an exception on error, unless
            C{bIgnoreExceptions} is C{True}.
        """
    try:
        cls.drop_privileges(win32.SE_DEBUG_NAME)
        return True
    except Exception:
        if not bIgnoreExceptions:
            raise
    return False