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
def helperFormat(self, f):
    r = Reader(bytes=_pngsuite['basn0g01'])
    o = BytesIO()

    def newchunks():
        for chunk in r.chunks():
            yield f(chunk)
    write_chunks(o, newchunks())
    r = Reader(bytes=o.getvalue())
    return list(r.asDirect()[2])