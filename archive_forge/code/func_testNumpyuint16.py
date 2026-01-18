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
def testNumpyuint16(self):
    """numpy uint16."""
    try:
        import numpy
    except ImportError:
        sys.stderr.write('skipping numpy test\n')
        return
    rows = [map(numpy.uint16, range(0, 65536, 21845))]
    b = topngbytes('numpyuint16.png', rows, 4, 1, greyscale=True, alpha=False, bitdepth=16)