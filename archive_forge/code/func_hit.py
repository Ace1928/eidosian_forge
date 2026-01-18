from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def hit(self, event):
    """
        Notify a breakpoint that it's been hit.

        This triggers the corresponding state transition and sets the
        C{breakpoint} property of the given L{Event} object.

        @see: L{disable}, L{enable}, L{one_shot}, L{running}

        @type  event: L{Event}
        @param event: Debug event to handle (depends on the breakpoint type).

        @raise AssertionError: Disabled breakpoints can't be hit.
        """
    aProcess = event.get_process()
    aThread = event.get_thread()
    state = self.get_state()
    event.breakpoint = self
    if state == self.ENABLED:
        self.running(aProcess, aThread)
    elif state == self.RUNNING:
        self.enable(aProcess, aThread)
    elif state == self.ONESHOT:
        self.disable(aProcess, aThread)
    elif state == self.DISABLED:
        msg = 'Hit a disabled breakpoint at address %s'
        msg = msg % HexDump.address(self.get_address())
        warnings.warn(msg, BreakpointWarning)