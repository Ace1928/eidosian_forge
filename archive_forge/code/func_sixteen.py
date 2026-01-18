from __future__ import absolute_import
import sys
import time
import serial
from serial.serialutil import  to_bytes
def sixteen(data):
    """    yield tuples of hex and ASCII display in multiples of 16. Includes a
    space after 8 bytes and (None, None) after 16 bytes and at the end.
    """
    n = 0
    for b in serial.iterbytes(data):
        yield ('{:02X} '.format(ord(b)), b.decode('ascii') if b' ' <= b < b'\x7f' else '.')
        n += 1
        if n == 8:
            yield (' ', '')
        elif n >= 16:
            yield (None, None)
            n = 0
    if n > 0:
        while n < 16:
            n += 1
            if n == 8:
                yield (' ', '')
            yield ('   ', ' ')
        yield (None, None)