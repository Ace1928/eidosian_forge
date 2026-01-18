from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexInput, HexDump
from winappdbg.util import PathOperations
import os
import warnings
import traceback
def clear_modules(self):
    """
        Clears the modules snapshot.
        """
    for aModule in compat.itervalues(self.__moduleDict):
        aModule.clear()
    self.__moduleDict = dict()