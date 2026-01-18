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
def seqtobytes(s):
    """Convert a sequence of integers to a *bytes* instance.  Good for
    plastering over Python 2 / Python 3 cracks.
    """
    return strtobytes(''.join((chr(x) for x in s)))