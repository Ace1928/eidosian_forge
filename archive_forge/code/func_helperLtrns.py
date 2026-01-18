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
def helperLtrns(self, transparent):
    """Helper used by :meth:`testLtrns*`."""
    pixels = zip([0, 56, 76, 84, 92, 64, 56, 0])
    o = BytesIO()
    w = Writer(8, 8, greyscale=True, bitdepth=1, transparent=transparent)
    w.write_packed(o, pixels)
    r = Reader(bytes=o.getvalue())
    x, y, pixels, meta = r.asDirect()
    self.assertTrue(meta['alpha'])
    self.assertTrue(meta['greyscale'])
    self.assertEqual(meta['bitdepth'], 1)