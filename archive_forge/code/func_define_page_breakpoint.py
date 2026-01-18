from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def define_page_breakpoint(self, dwProcessId, address, pages=1, condition=True, action=None):
    """
        Creates a disabled page breakpoint at the given address.

        @see:
            L{has_page_breakpoint},
            L{get_page_breakpoint},
            L{enable_page_breakpoint},
            L{enable_one_shot_page_breakpoint},
            L{disable_page_breakpoint},
            L{erase_page_breakpoint}

        @type  dwProcessId: int
        @param dwProcessId: Process global ID.

        @type  address: int
        @param address: Memory address of the first page to watch.

        @type  pages: int
        @param pages: Number of pages to watch.

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

        @rtype:  L{PageBreakpoint}
        @return: The page breakpoint object.
        """
    process = self.system.get_process(dwProcessId)
    bp = PageBreakpoint(address, pages, condition, action)
    begin = bp.get_address()
    end = begin + bp.get_size()
    address = begin
    pageSize = MemoryAddresses.pageSize
    while address < end:
        key = (dwProcessId, address)
        if key in self.__pageBP:
            msg = 'Already exists (PID %d) : %r'
            msg = msg % (dwProcessId, self.__pageBP[key])
            raise KeyError(msg)
        address = address + pageSize
    address = begin
    while address < end:
        key = (dwProcessId, address)
        self.__pageBP[key] = bp
        address = address + pageSize
    return bp