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
def testNumpybool(self):
    """numpy bool."""
    try:
        import numpy
    except ImportError:
        sys.stderr.write('skipping numpy test\n')
        return
    rows = [map(numpy.bool, [0, 1])]
    b = topngbytes('numpybool.png', rows, 2, 1, greyscale=True, alpha=False, bitdepth=1)