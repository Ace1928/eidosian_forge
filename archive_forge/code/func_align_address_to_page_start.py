import sys
import os
import ctypes
import optparse
from winappdbg import win32
from winappdbg import compat
@classmethod
def align_address_to_page_start(cls, address):
    """
        Align the given address to the start of the page it occupies.

        @type  address: int
        @param address: Memory address.

        @rtype:  int
        @return: Aligned memory address.
        """
    return address - address % cls.pageSize