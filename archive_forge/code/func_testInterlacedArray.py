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
def testInterlacedArray(self):
    """Test that reading an interlaced PNG yields each row as an
        array."""
    r = Reader(bytes=_pngsuite['basi0g08'])
    list(r.read()[2])[0].tostring