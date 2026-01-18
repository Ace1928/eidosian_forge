import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
@staticmethod
def dump_flags(efl):
    """
        Dump the x86 processor flags.
        The output mimics that of the WinDBG debugger.
        Used by L{dump_registers}.

        @type  efl: int
        @param efl: Value of the eFlags register.

        @rtype:  str
        @return: Text suitable for logging.
        """
    if efl is None:
        return ''
    efl_dump = 'iopl=%1d' % ((efl & 12288) >> 12)
    if efl & 1048576:
        efl_dump += ' vip'
    else:
        efl_dump += '    '
    if efl & 524288:
        efl_dump += ' vif'
    else:
        efl_dump += '    '
    if efl & 2048:
        efl_dump += ' ov'
    else:
        efl_dump += ' no'
    if efl & 1024:
        efl_dump += ' dn'
    else:
        efl_dump += ' up'
    if efl & 512:
        efl_dump += ' ei'
    else:
        efl_dump += ' di'
    if efl & 128:
        efl_dump += ' ng'
    else:
        efl_dump += ' pl'
    if efl & 64:
        efl_dump += ' zr'
    else:
        efl_dump += ' nz'
    if efl & 16:
        efl_dump += ' ac'
    else:
        efl_dump += ' na'
    if efl & 4:
        efl_dump += ' pe'
    else:
        efl_dump += ' po'
    if efl & 1:
        efl_dump += ' cy'
    else:
        efl_dump += ' nc'
    return efl_dump