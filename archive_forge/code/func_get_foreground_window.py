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
@staticmethod
def get_foreground_window():
    """
        @rtype:  L{Window}
        @return: Returns the foreground window.
        @raise WindowsError: An error occured while processing this request.
        """
    return Window(win32.GetForegroundWindow())