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
def cont(self, event=None):
    """
        Resumes execution after processing a debug event.

        @see: dispatch(), loop(), wait()

        @type  event: L{Event}
        @param event: (Optional) Event object returned by L{wait}.

        @raise WindowsError: Raises an exception on error.
        """
    if event is None:
        event = self.lastEvent
    if not event:
        return
    dwProcessId = event.get_pid()
    dwThreadId = event.get_tid()
    dwContinueStatus = event.continueStatus
    if self.is_debugee(dwProcessId):
        try:
            if self.system.has_process(dwProcessId):
                aProcess = self.system.get_process(dwProcessId)
            else:
                aProcess = Process(dwProcessId)
            aProcess.flush_instruction_cache()
        except WindowsError:
            pass
        win32.ContinueDebugEvent(dwProcessId, dwThreadId, dwContinueStatus)
    if event == self.lastEvent:
        self.lastEvent = None