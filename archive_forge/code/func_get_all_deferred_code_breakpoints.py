from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def get_all_deferred_code_breakpoints(self):
    """
        Returns a list of deferred code breakpoints.

        @rtype:  tuple of (int, str, callable, bool)
        @return: Tuple containing the following elements:
             - Process ID where to set the breakpoint.
             - Label pointing to the address where to set the breakpoint.
             - Action callback for the breakpoint.
             - C{True} of the breakpoint is one-shot, C{False} otherwise.
        """
    result = []
    for pid, deferred in compat.iteritems(self.__deferredBP):
        for label, (action, oneshot) in compat.iteritems(deferred):
            result.add((pid, label, action, oneshot))
    return result