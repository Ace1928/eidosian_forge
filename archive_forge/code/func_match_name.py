from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexInput, HexDump
from winappdbg.util import PathOperations
import os
import warnings
import traceback
def match_name(self, name):
    """
        @rtype:  bool
        @return:
            C{True} if the given name could refer to this module.
            It may not be exactly the same returned by L{get_name}.
        """
    my_name = self.get_name().lower()
    if name.lower() == my_name:
        return True
    try:
        base = HexInput.integer(name)
    except ValueError:
        base = None
    if base is not None and base == self.get_base():
        return True
    modName = self.__filename_to_modname(name)
    if modName.lower() == my_name:
        return True
    return False