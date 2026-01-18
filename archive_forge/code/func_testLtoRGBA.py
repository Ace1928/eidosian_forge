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
def testLtoRGBA(self):
    """asRGBA() on grey source."""
    r = Reader(bytes=_pngsuite['basi0g08'])
    x, y, pixels, meta = r.asRGBA()
    row9 = list(list(pixels)[9])
    self.assertEqual(row9[0:8], [222, 222, 222, 255, 221, 221, 221, 255])