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
def remove_key(self, key):
    """
        Removes the given key from the set of known keys.

        @type  key: L{Crash} key.
        @param key: Key to remove.
        """
    del self.__keys[key]