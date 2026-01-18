from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexInput, HexDump
from winappdbg.util import PathOperations
import os
import warnings
import traceback
def get_module_by_name(self, modName):
    """
        @type  modName: int
        @param modName:
            Name of the module to look for, as returned by L{Module.get_name}.
            If two or more modules with the same name are loaded, only one
            of the matching modules is returned.

            You can also pass a full pathname to the DLL file.
            This works correctly even if two modules with the same name
            are loaded from different paths.

        @rtype:  L{Module}
        @return: C{Module} object that best matches the given name.
            Returns C{None} if no C{Module} can be found.
        """
    modName = modName.lower()
    if PathOperations.path_is_absolute(modName):
        for lib in self.iter_modules():
            if modName == lib.get_filename().lower():
                return lib
        return None
    modDict = [(lib.get_name(), lib) for lib in self.iter_modules()]
    modDict = dict(modDict)
    if modName in modDict:
        return modDict[modName]
    filepart, extpart = PathOperations.split_extension(modName)
    if filepart and extpart:
        if filepart in modDict:
            return modDict[filepart]
    try:
        baseAddress = HexInput.integer(modName)
    except ValueError:
        return None
    if self.has_module(baseAddress):
        return self.get_module(baseAddress)
    return None