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
def testPGMin(self):
    """Test that the command line tool can read PGM files."""

    def do():
        return _main(['testPGMin'])
    s = BytesIO()
    s.write(strtobytes('P5 2 2 3\n'))
    s.write(strtobytes('\x00\x01\x02\x03'))
    s.flush()
    s.seek(0)
    o = BytesIO()
    testWithIO(s, o, do)
    r = Reader(bytes=o.getvalue())
    x, y, pixels, meta = r.read()
    self.assertTrue(r.greyscale)
    self.assertEqual(r.bitdepth, 2)