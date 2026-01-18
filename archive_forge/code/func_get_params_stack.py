from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def get_params_stack(self, tid):
    """
        Returns the parameters found in the stack each time the hooked function
        was called by this thread and hasn't returned yet.

        @type  tid: int
        @param tid: Thread global ID.

        @rtype:  list of tuple( arg, arg, arg... )
        @return: List of argument tuples.
        """
    try:
        stack = self.__paramStack[tid]
    except KeyError:
        msg = 'Hooked function was not called from thread %d'
        raise KeyError(msg % tid)
    return stack