from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def get_stack_trace_with_labels(self, depth=16, bMakePretty=True):
    """
        Tries to get a stack trace for the current function.
        Only works for functions with standard prologue and epilogue.

        @type  depth: int
        @param depth: Maximum depth of stack trace.

        @type  bMakePretty: bool
        @param bMakePretty:
            C{True} for user readable labels,
            C{False} for labels that can be passed to L{Process.resolve_label}.

            "Pretty" labels look better when producing output for the user to
            read, while pure labels are more useful programatically.

        @rtype:  tuple of tuple( int, int, str )
        @return: Stack trace of the thread as a tuple of
            ( return address, frame pointer label ).

        @raise WindowsError: Raises an exception on error.
        """
    try:
        trace = self.__get_stack_trace(depth, True, bMakePretty)
    except Exception:
        trace = ()
    if not trace:
        trace = self.__get_stack_trace_manually(depth, True, bMakePretty)
    return trace