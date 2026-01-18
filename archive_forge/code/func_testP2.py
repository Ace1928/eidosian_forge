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
def testP2(self):
    """2-bit palette."""
    a = (255, 255, 255)
    b = (200, 120, 120)
    c = (50, 99, 50)
    w = Writer(1, 4, bitdepth=2, palette=[a, b, c])
    f = BytesIO()
    w.write_array(f, array('B', (0, 1, 1, 2)))
    r = Reader(bytes=f.getvalue())
    x, y, pixels, meta = r.asRGB8()
    self.assertEqual(x, 1)
    self.assertEqual(y, 4)
    self.assertEqual(list(pixels), map(list, [a, b, b, c]))