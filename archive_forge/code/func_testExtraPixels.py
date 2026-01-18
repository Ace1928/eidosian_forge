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
def testExtraPixels(self):
    """Test file that contains too many pixels."""

    def eachchunk(chunk):
        if chunk[0] != 'IDAT':
            return chunk
        data = zlib.decompress(chunk[1])
        data += strtobytes('\x00garbage')
        data = zlib.compress(data)
        chunk = (chunk[0], data)
        return chunk
    self.assertRaises(FormatError, self.helperFormat, eachchunk)