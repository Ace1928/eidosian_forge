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
def testPackedIter(self):
    """Test iterator for row when using write_packed.

        Indicative for Issue 47.
        """
    w = Writer(16, 2, greyscale=True, alpha=False, bitdepth=1)
    o = BytesIO()
    w.write_packed(o, [itertools.chain([10], [170]), itertools.chain([15], [255])])
    r = Reader(bytes=o.getvalue())
    x, y, pixels, info = r.asDirect()
    pixels = list(pixels)
    self.assertEqual(len(pixels), 2)
    self.assertEqual(len(pixels[0]), 16)