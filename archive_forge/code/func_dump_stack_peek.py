import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
@staticmethod
def dump_stack_peek(data, separator=' ', width=16, arch=None):
    """
        Dump data from pointers guessed within the given stack dump.

        @type  data: str
        @param data: Dictionary mapping stack offsets to the data they point to.

        @type  separator: str
        @param separator:
            Separator between the hexadecimal representation of each character.

        @type  width: int
        @param width:
            (Optional) Maximum number of characters to convert per text line.
            This value is also used for padding.

        @type  arch: str
        @param arch: Architecture of the machine whose registers were dumped.
            Defaults to the current architecture.

        @rtype:  str
        @return: Text suitable for logging.
        """
    if data is None:
        return ''
    if arch is None:
        arch = win32.arch
    pointers = compat.keys(data)
    pointers.sort()
    result = ''
    if pointers:
        if arch == win32.ARCH_I386:
            spreg = 'esp'
        elif arch == win32.ARCH_AMD64:
            spreg = 'rsp'
        else:
            spreg = 'STACK'
        tag_fmt = '[%s+0x%%.%dx]' % (spreg, len('%x' % pointers[-1]))
        for offset in pointers:
            dumped = HexDump.hexline(data[offset], separator, width)
            tag = tag_fmt % offset
            result += '%s -> %s\n' % (tag, dumped)
    return result