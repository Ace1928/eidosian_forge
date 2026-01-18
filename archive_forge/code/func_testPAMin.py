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
def testPAMin(self):
    """Test that the command line tool can read PAM file."""

    def do():
        return _main(['testPAMin'])
    s = BytesIO()
    s.write(strtobytes('P7\nWIDTH 3\nHEIGHT 1\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n'))
    flat = [255, 0, 0, 255, 0, 255, 0, 120, 0, 0, 255, 30]
    asbytes = seqtobytes(flat)
    s.write(asbytes)
    s.flush()
    s.seek(0)
    o = BytesIO()
    testWithIO(s, o, do)
    r = Reader(bytes=o.getvalue())
    x, y, pixels, meta = r.read()
    self.assertTrue(r.alpha)
    self.assertTrue(not r.greyscale)
    self.assertEqual(list(itertools.chain(*pixels)), flat)