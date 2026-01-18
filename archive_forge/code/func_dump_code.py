import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
@staticmethod
def dump_code(disassembly, pc=None, bLowercase=True, bits=None):
    """
        Dump a disassembly. Optionally mark where the program counter is.

        @type  disassembly: list of tuple( int, int, str, str )
        @param disassembly: Disassembly dump as returned by
            L{Process.disassemble} or L{Thread.disassemble_around_pc}.

        @type  pc: int
        @param pc: (Optional) Program counter.

        @type  bLowercase: bool
        @param bLowercase: (Optional) If C{True} convert the code to lowercase.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexDump.address_size}

        @rtype:  str
        @return: Text suitable for logging.
        """
    if not disassembly:
        return ''
    table = Table(sep=' | ')
    for addr, size, code, dump in disassembly:
        if bLowercase:
            code = code.lower()
        if addr == pc:
            addr = ' * %s' % HexDump.address(addr, bits)
        else:
            addr = '   %s' % HexDump.address(addr, bits)
        table.addRow(addr, dump, code)
    table.justify(1, 1)
    return table.getOutput()