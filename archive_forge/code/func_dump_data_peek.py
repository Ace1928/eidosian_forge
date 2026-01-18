import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
@staticmethod
def dump_data_peek(data, base=0, separator=' ', width=16, bits=None):
    """
        Dump data from pointers guessed within the given binary data.

        @type  data: str
        @param data: Dictionary mapping offsets to the data they point to.

        @type  base: int
        @param base: Base offset.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexDump.address_size}

        @rtype:  str
        @return: Text suitable for logging.
        """
    if data is None:
        return ''
    pointers = compat.keys(data)
    pointers.sort()
    result = ''
    for offset in pointers:
        dumped = HexDump.hexline(data[offset], separator, width)
        address = HexDump.address(base + offset, bits)
        result += '%s -> %s\n' % (address, dumped)
    return result