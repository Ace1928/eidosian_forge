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
def iterdecomp(idat):
    """Iterator that yields decompressed strings.  `idat` should
            be an iterator that yields the ``IDAT`` chunk data.
            """
    d = zlib.decompressobj()
    for data in idat:
        yield array('B', d.decompress(data))
    yield array('B', d.flush())