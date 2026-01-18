from __future__ import print_function   # This version of olefile requires Python 2.7 or 3.5+.
import io
import sys
import struct, array, os.path, datetime, logging, warnings, traceback
def i16(c, o=0):
    """
    Converts a 2-bytes (16 bits) string to an integer.

    :param c: string containing bytes to convert
    :param o: offset of bytes to convert in string
    """
    return struct.unpack('<H', c[o:o + 2])[0]