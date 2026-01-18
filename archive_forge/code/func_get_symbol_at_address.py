from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexInput, HexDump
from winappdbg.util import PathOperations
import os
import warnings
import traceback
def get_symbol_at_address(self, address):
    """
        Tries to find the closest matching symbol for the given address.

        @type  address: int
        @param address: Memory address to query.

        @rtype: None or tuple( str, int, int )
        @return: Returns a tuple consisting of:
             - Name
             - Address
             - Size (in bytes)
            Returns C{None} if no symbol could be matched.
        """
    found = None
    for SymbolName, SymbolAddress, SymbolSize in self.iter_symbols():
        if SymbolAddress > address:
            continue
        if SymbolAddress == address:
            found = (SymbolName, SymbolAddress, SymbolSize)
            break
        if SymbolAddress < address:
            if found and address - found[1] < address - SymbolAddress:
                continue
            else:
                found = (SymbolName, SymbolAddress, SymbolSize)
    return found