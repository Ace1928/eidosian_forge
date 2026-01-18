from winappdbg import win32
from winappdbg import compat
from winappdbg.system import System
from winappdbg.textio import HexDump, CrashDump
from winappdbg.util import StaticClass, MemoryAddresses, PathOperations
import sys
import os
import time
import zlib
import warnings
@property
def pc(self):
    """
        Value of the program counter register.

        @rtype:  int
        """
    try:
        return self.registers['Eip']
    except KeyError:
        return self.registers['Rip']