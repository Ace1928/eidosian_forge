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
def marshall_value(self, value, storeMemoryMap=False):
    """
        Marshalls a Crash object to be used in the database.
        By default the C{memoryMap} member is B{NOT} stored here.

        @warning: Setting the C{storeMemoryMap} argument to C{True} can lead to
            a severe performance penalty!

        @type  value: L{Crash}
        @param value: Object to convert.

        @type  storeMemoryMap: bool
        @param storeMemoryMap: C{True} to store the memory map, C{False}
            otherwise.

        @rtype:  str
        @return: Converted object.
        """
    if hasattr(value, 'memoryMap'):
        crash = value
        memoryMap = crash.memoryMap
        try:
            crash.memoryMap = None
            if storeMemoryMap and memoryMap is not None:
                crash.memoryMap = list(memoryMap)
            if self.optimizeValues:
                value = pickle.dumps(crash, protocol=HIGHEST_PROTOCOL)
                value = optimize(value)
            else:
                value = pickle.dumps(crash, protocol=0)
        finally:
            crash.memoryMap = memoryMap
            del memoryMap
            del crash
    if self.compressValues:
        value = zlib.compress(value, zlib.Z_BEST_COMPRESSION)
    if self.escapeValues:
        value = value.encode('hex')
    if self.binaryValues:
        value = buffer(value)
    return value