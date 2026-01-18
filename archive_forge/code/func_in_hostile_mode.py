import sys
from winappdbg import win32
from winappdbg.system import System
from winappdbg.process import Process
from winappdbg.thread import Thread
from winappdbg.module import Module
from winappdbg.window import Window
from winappdbg.breakpoint import _BreakpointContainer, CodeBreakpoint
from winappdbg.event import Event, EventHandler, EventDispatcher, EventFactory
from winappdbg.interactive import ConsoleDebugger
import warnings
def in_hostile_mode(self):
    """
        Determine if we're in hostile mode (anti-anti-debug).

        @rtype:  bool
        @return: C{True} if this C{Debug} instance was started in hostile mode,
            C{False} otherwise.
        """
    return self.__bHostileCode