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
def is_debugee(self, dwProcessId):
    """
        Determine if the debugger is debugging the given process.

        @see: L{is_debugee_attached}, L{is_debugee_started}

        @type  dwProcessId: int
        @param dwProcessId: Process global ID.

        @rtype:  bool
        @return: C{True} if the given process is being debugged
            by this L{Debug} instance.
        """
    return self.is_debugee_attached(dwProcessId) or self.is_debugee_started(dwProcessId)