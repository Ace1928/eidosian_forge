from __future__ import print_function   # This version of olefile requires Python 2.7 or 3.5+.
import io
import sys
import struct, array, os.path, datetime, logging, warnings, traceback
def sect2array(self, sect):
    """
        convert a sector to an array of 32 bits unsigned integers,
        swapping bytes on big endian CPUs such as PowerPC (old Macs)
        """
    a = array.array(UINT32, sect)
    if sys.byteorder == 'big':
        a.byteswap()
    return a