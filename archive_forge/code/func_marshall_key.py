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
def marshall_key(self, key):
    """
        Marshalls a Crash key to be used in the database.

        @see: L{__init__}

        @type  key: L{Crash} key.
        @param key: Key to convert.

        @rtype:  str or buffer
        @return: Converted key.
        """
    if key in self.__keys:
        return self.__keys[key]
    skey = pickle.dumps(key, protocol=0)
    if self.compressKeys:
        skey = zlib.compress(skey, zlib.Z_BEST_COMPRESSION)
    if self.escapeKeys:
        skey = skey.encode('hex')
    if self.binaryKeys:
        skey = buffer(skey)
    self.__keys[key] = skey
    return skey