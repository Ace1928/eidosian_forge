import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
@classmethod
def hexblock_word(cls, data, address=None, bits=None, separator=' ', width=8):
    """
        Dump a block of hexadecimal WORDs from binary data.

        @type  data: str
        @param data: Binary data.

        @type  address: str
        @param address: Memory address where the data was read from.

        @type  bits: int
        @param bits:
            (Optional) Number of bits of the target architecture.
            The default is platform dependent. See: L{HexDump.address_size}

        @type  separator: str
        @param separator:
            Separator between the hexadecimal representation of each WORD.

        @type  width: int
        @param width:
            (Optional) Maximum number of WORDs to convert per text line.

        @rtype:  str
        @return: Multiline output text.
        """
    return cls.hexblock_cb(cls.hexa_word, data, address, bits, width * 2, cb_kwargs={'separator': separator})