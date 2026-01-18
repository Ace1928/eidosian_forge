from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def define_code_breakpoint(self, dwProcessId, address, condition=True, action=None):
    """
        Creates a disabled code breakpoint at the given address.

        @see:
            L{has_code_breakpoint},
            L{get_code_breakpoint},
            L{enable_code_breakpoint},
            L{enable_one_shot_code_breakpoint},
            L{disable_code_breakpoint},
            L{erase_code_breakpoint}

        @type  dwProcessId: int
        @param dwProcessId: Process global ID.

        @type  address: int
        @param address: Memory address of the code instruction to break at.

        @type  condition: function
        @param condition: (Optional) Condition callback function.

            The callback signature is::

                def condition_callback(event):
                    return True     # returns True or False

            Where B{event} is an L{Event} object,
            and the return value is a boolean
            (C{True} to dispatch the event, C{False} otherwise).

        @type  action: function
        @param action: (Optional) Action callback function.
            If specified, the event is handled by this callback instead of
            being dispatched normally.

            The callback signature is::

                def action_callback(event):
                    pass        # no return value

            Where B{event} is an L{Event} object,
            and the return value is a boolean
            (C{True} to dispatch the event, C{False} otherwise).

        @rtype:  L{CodeBreakpoint}
        @return: The code breakpoint object.
        """
    process = self.system.get_process(dwProcessId)
    bp = CodeBreakpoint(address, condition, action)
    key = (dwProcessId, bp.get_address())
    if key in self.__codeBP:
        msg = 'Already exists (PID %d) : %r'
        raise KeyError(msg % (dwProcessId, self.__codeBP[key]))
    self.__codeBP[key] = bp
    return bp