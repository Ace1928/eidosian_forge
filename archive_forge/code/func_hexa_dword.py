import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
@staticmethod
def hexa_dword(data, separator=' '):
    """
        Convert binary data to a string of hexadecimal DWORDs.

        @type  data: str
        @param data: Binary data.

        @type  separator: str
        @param separator:
            Separator between the hexadecimal representation of each DWORD.

        @rtype:  str
        @return: Hexadecimal representation.
        """
    if len(data) & 3 != 0:
        data += '\x00' * (4 - (len(data) & 3))
    return separator.join(['%.8x' % struct.unpack('<L', data[i:i + 4])[0] for i in compat.xrange(0, len(data), 4)])