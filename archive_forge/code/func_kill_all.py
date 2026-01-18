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
def kill_all(self, bIgnoreExceptions=False):
    """
        Kills from all processes currently being debugged.

        @type  bIgnoreExceptions: bool
        @param bIgnoreExceptions: C{True} to ignore any exceptions that may be
            raised when killing each process. C{False} to stop and raise an
            exception when encountering an error.

        @raise WindowsError: Raises an exception on error, unless
            C{bIgnoreExceptions} is C{True}.
        """
    for pid in self.get_debugee_pids():
        self.kill(pid, bIgnoreExceptions=bIgnoreExceptions)