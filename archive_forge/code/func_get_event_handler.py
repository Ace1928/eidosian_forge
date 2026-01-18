from winappdbg import win32
from winappdbg import compat
from winappdbg.win32 import FileHandle, ProcessHandle, ThreadHandle
from winappdbg.breakpoint import ApiHook
from winappdbg.module import Module
from winappdbg.thread import Thread
from winappdbg.process import Process
from winappdbg.textio import HexDump
from winappdbg.util import StaticClass, PathOperations
import sys
import ctypes
import warnings
import traceback
def get_event_handler(self):
    """
        Get the event handler.

        @see: L{set_event_handler}

        @rtype:  L{EventHandler}
        @return: Current event handler object, or C{None}.
        """
    return self.__eventHandler