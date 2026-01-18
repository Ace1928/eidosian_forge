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
def testRGBtoRGBA(self):
    """asRGBA8() on colour type 2 source."""
    r = Reader(bytes=_pngsuite['basn2c08'])
    x, y, pixels, meta = r.asRGBA8()
    row9 = list(pixels)[9]
    self.assertEqual(row9[0:8], [255, 223, 255, 255, 255, 222, 255, 255])