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
def briefReport(self):
    """
        @rtype:  str
        @return: Short description of the event.
        """
    if self.exceptionCode is not None:
        if self.exceptionCode == win32.EXCEPTION_BREAKPOINT:
            if self.isOurBreakpoint:
                what = 'Breakpoint hit'
            elif self.isSystemBreakpoint:
                what = 'System breakpoint hit'
            else:
                what = 'Assertion failed'
        elif self.exceptionDescription:
            what = self.exceptionDescription
        elif self.exceptionName:
            what = self.exceptionName
        else:
            what = 'Exception %s' % HexDump.integer(self.exceptionCode, self.bits)
        if self.firstChance:
            chance = 'first'
        else:
            chance = 'second'
        if self.exceptionLabel:
            where = self.exceptionLabel
        elif self.exceptionAddress:
            where = HexDump.address(self.exceptionAddress, self.bits)
        elif self.labelPC:
            where = self.labelPC
        else:
            where = HexDump.address(self.pc, self.bits)
        msg = '%s (%s chance) at %s' % (what, chance, where)
    elif self.debugString is not None:
        if self.labelPC:
            where = self.labelPC
        else:
            where = HexDump.address(self.pc, self.bits)
        msg = 'Debug string from %s: %r' % (where, self.debugString)
    else:
        if self.labelPC:
            where = self.labelPC
        else:
            where = HexDump.address(self.pc, self.bits)
        msg = '%s (%s) at %s' % (self.eventName, HexDump.integer(self.eventCode, self.bits), where)
    return msg