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
def get_debugee_pids(self):
    """
        @rtype:  list( int... )
        @return: Global IDs of processes being debugged.
        """
    return list(self.__attachedDebugees) + list(self.__startedDebugees)