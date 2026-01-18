from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexInput, HexDump
from winappdbg.util import PathOperations
import os
import warnings
import traceback
def get_module_at_address(self, address):
    """
        @type  address: int
        @param address: Memory address to query.

        @rtype:  L{Module}
        @return: C{Module} object that best matches the given address.
            Returns C{None} if no C{Module} can be found.
        """
    bases = self.get_module_bases()
    bases.sort()
    bases.append(long(18446744073709551616))
    if address >= bases[0]:
        i = 0
        max_i = len(bases) - 1
        while i < max_i:
            begin, end = bases[i:i + 2]
            if begin <= address < end:
                module = self.get_module(begin)
                here = module.is_address_here(address)
                if here is False:
                    break
                else:
                    return module
            i = i + 1
    return None