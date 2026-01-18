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
def interleave_planes(ipixels, apixels, ipsize, apsize):
    """
    Interleave (colour) planes, e.g. RGB + A = RGBA.

    Return an array of pixels consisting of the `ipsize` elements of data
    from each pixel in `ipixels` followed by the `apsize` elements of data
    from each pixel in `apixels`.  Conventionally `ipixels` and
    `apixels` are byte arrays so the sizes are bytes, but it actually
    works with any arrays of the same type.  The returned array is the
    same type as the input arrays which should be the same type as each other.
    """
    itotal = len(ipixels)
    atotal = len(apixels)
    newtotal = itotal + atotal
    newpsize = ipsize + apsize
    out = array(ipixels.typecode)
    out.extend(ipixels)
    out.extend(apixels)
    for i in range(ipsize):
        out[i:newtotal:newpsize] = ipixels[i:itotal:ipsize]
    for i in range(apsize):
        out[i + ipsize:newtotal:newpsize] = apixels[i:atotal:apsize]
    return out