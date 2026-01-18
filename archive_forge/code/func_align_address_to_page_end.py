import sys
import os
import ctypes
import optparse
from winappdbg import win32
from winappdbg import compat
@classmethod
def align_address_to_page_end(cls, address):
    """
        Align the given address to the end of the page it occupies.
        That is, to point to the start of the next page.

        @type  address: int
        @param address: Memory address.

        @rtype:  int
        @return: Aligned memory address.
        """
    return address + cls.pageSize - address % cls.pageSize