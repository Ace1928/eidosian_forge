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
def testNumpyuint8(self):
    """numpy uint8."""
    try:
        import numpy
    except ImportError:
        sys.stderr.write('skipping numpy test\n')
        return
    rows = [map(numpy.uint8, range(0, 256, 85))]
    b = topngbytes('numpyuint8.png', rows, 4, 1, greyscale=True, alpha=False, bitdepth=8)