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
def testTrnsArray(self):
    """Test that reading a type 2 PNG with tRNS chunk yields each
        row as an array (using asDirect)."""
    r = Reader(bytes=_pngsuite['tbrn2c08'])
    list(r.asDirect()[2])[0].tostring