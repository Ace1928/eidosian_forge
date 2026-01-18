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
def notesReport(self):
    """
        @rtype:  str
        @return: All notes, merged and formatted for a report.
        """
    msg = ''
    if self.notes:
        for n in self.notes:
            n = n.strip('\n')
            if '\n' in n:
                n = n.strip('\n')
                msg += ' * %s\n' % n.pop(0)
                for x in n:
                    msg += '   %s\n' % x
            else:
                msg += ' * %s\n' % n
    return msg