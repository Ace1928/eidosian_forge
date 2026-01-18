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
def testFlat(self):
    """Test read_flat."""
    import hashlib
    r = Reader(bytes=_pngsuite['basn0g02'])
    x, y, pixel, meta = r.read_flat()
    d = hashlib.md5(seqtobytes(pixel)).digest()
    self.assertEqual(_enhex(d), '255cd971ab8cd9e7275ff906e5041aa0')