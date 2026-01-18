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
def testL2(self):
    """Also tests asRGB8."""
    w = Writer(1, 4, greyscale=True, bitdepth=2)
    f = BytesIO()
    w.write_array(f, array('B', range(4)))
    r = Reader(bytes=f.getvalue())
    x, y, pixels, meta = r.asRGB8()
    self.assertEqual(x, 1)
    self.assertEqual(y, 4)
    for i, row in enumerate(pixels):
        self.assertEqual(len(row), 3)
        self.assertEqual(list(row), [85 * i] * 3)