from the public API.  This format is called packed.  When packed,
import io
import itertools
import math
import operator
import struct
import sys
import zlib
import warnings
from array import array
from functools import reduce
from pygame.tests.test_utils import tostring
fromarray = from_array
import tempfile
import unittest
def iterstraight(self, raw):
    """Iterator that undoes the effect of filtering, and yields each
        row in serialised format (as a sequence of bytes).  Assumes input
        is straightlaced.  `raw` should be an iterable that yields the
        raw bytes in chunks of arbitrary size."""
    rb = self.row_bytes
    a = array('B')
    recon = None
    for some in raw:
        a.extend(some)
        while len(a) >= rb + 1:
            filter_type = a[0]
            scanline = a[1:rb + 1]
            del a[:rb + 1]
            recon = self.undo_filter(filter_type, scanline, recon)
            yield recon
    if len(a) != 0:
        raise FormatError('Wrong size for decompressed IDAT chunk.')
    assert len(a) == 0