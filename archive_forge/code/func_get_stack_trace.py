from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def get_stack_trace(self, depth=16):
    """
        Tries to get a stack trace for the current function.
        Only works for functions with standard prologue and epilogue.

        @type  depth: int
        @param depth: Maximum depth of stack trace.

        @rtype:  tuple of tuple( int, int, str )
        @return: Stack trace of the thread as a tuple of
            ( return address, frame pointer address, module filename ).

        @raise WindowsError: Raises an exception on error.
        """
    try:
        trace = self.__get_stack_trace(depth, False)
    except Exception:
        import traceback
        traceback.print_exc()
        trace = ()
    if not trace:
        trace = self.__get_stack_trace_manually(depth, False)
    return trace