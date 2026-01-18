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
def sp(self):
    """
        Value of the stack pointer register.

        @rtype:  int
        """
    try:
        return self.registers['Esp']
    except KeyError:
        return self.registers['Rsp']