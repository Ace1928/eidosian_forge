import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
@staticmethod
def dump_stack_trace(stack_trace, bits=None):
    """
        Dump a stack trace, as returned by L{Thread.get_stack_trace} with the
        C{bUseLabels} parameter set to C{False}.

        @type  stack_trace: list( int, int, str )
        @param stack_trace: Stack trace as a list of tuples of
            ( return address, frame pointer, module filename )

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexDump.address_size}

        @rtype:  str
        @return: Text suitable for logging.
        """
    if not stack_trace:
        return ''
    table = Table()
    table.addRow('Frame', 'Origin', 'Module')
    for fp, ra, mod in stack_trace:
        fp_d = HexDump.address(fp, bits)
        ra_d = HexDump.address(ra, bits)
        table.addRow(fp_d, ra_d, mod)
    return table.getOutput()