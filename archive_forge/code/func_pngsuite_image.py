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
def pngsuite_image(name):
    """
        Create a test image by reading an internal copy of the files
        from the PngSuite.  Returned in flat row flat pixel format.
        """
    if name not in _pngsuite:
        raise NotImplementedError(f'cannot find PngSuite file {name} (use -L for a list)')
    r = Reader(bytes=_pngsuite[name])
    w, h, pixels, meta = r.asDirect()
    assert w == h
    if meta['greyscale'] and meta['alpha'] and (meta['bitdepth'] < 8):
        factor = 255 // (2 ** meta['bitdepth'] - 1)

        def rescale(data):
            for row in data:
                yield map(factor.__mul__, row)
        pixels = rescale(pixels)
        meta['bitdepth'] = 8
    arraycode = 'BH'[meta['bitdepth'] > 8]
    return (w, array(arraycode, itertools.chain(*pixels)), meta)