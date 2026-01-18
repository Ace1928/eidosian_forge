from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexInput, HexDump
from winappdbg.util import PathOperations
import os
import warnings
import traceback
def get_label_at_address(self, address, offset=None):
    """
        Creates a label from the given memory address.

        @warning: This method uses the name of the nearest currently loaded
            module. If that module is unloaded later, the label becomes
            impossible to resolve.

        @type  address: int
        @param address: Memory address.

        @type  offset: None or int
        @param offset: (Optional) Offset value.

        @rtype:  str
        @return: Label pointing to the given address.
        """
    if offset:
        address = address + offset
    modobj = self.get_module_at_address(address)
    if modobj:
        label = modobj.get_label_at_address(address)
    else:
        label = self.parse_label(None, None, address)
    return label