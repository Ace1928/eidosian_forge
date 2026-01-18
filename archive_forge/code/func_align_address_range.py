import sys
import os
import ctypes
import optparse
from winappdbg import win32
from winappdbg import compat
@classmethod
def align_address_range(cls, begin, end):
    """
        Align the given address range to the start and end of the page(s) it occupies.

        @type  begin: int
        @param begin: Memory address of the beginning of the buffer.
            Use C{None} for the first legal address in the address space.

        @type  end: int
        @param end: Memory address of the end of the buffer.
            Use C{None} for the last legal address in the address space.

        @rtype:  tuple( int, int )
        @return: Aligned memory addresses.
        """
    if begin is None:
        begin = 0
    if end is None:
        end = win32.LPVOID(-1).value
    if end < begin:
        begin, end = (end, begin)
    begin = cls.align_address_to_page_start(begin)
    if end != cls.align_address_to_page_start(end):
        end = cls.align_address_to_page_end(end)
    return (begin, end)