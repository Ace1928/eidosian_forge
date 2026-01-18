import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
@staticmethod
def hexa_word(data, separator=' '):
    """
        Convert binary data to a string of hexadecimal WORDs.

        @type  data: str
        @param data: Binary data.

        @type  separator: str
        @param separator:
            Separator between the hexadecimal representation of each WORD.

        @rtype:  str
        @return: Hexadecimal representation.
        """
    if len(data) & 1 != 0:
        data += '\x00'
    return separator.join(['%.4x' % struct.unpack('<H', data[i:i + 2])[0] for i in compat.xrange(0, len(data), 2)])