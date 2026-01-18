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
def make_palette(self):
    """Create the byte sequences for a ``PLTE`` and if necessary a
        ``tRNS`` chunk.  Returned as a pair (*p*, *t*).  *t* will be
        ``None`` if no ``tRNS`` chunk is necessary.
        """
    p = array('B')
    t = array('B')
    for x in self.palette:
        p.extend(x[0:3])
        if len(x) > 3:
            t.append(x[3])
    p = tostring(p)
    t = tostring(t)
    if t:
        return (p, t)
    return (p, None)