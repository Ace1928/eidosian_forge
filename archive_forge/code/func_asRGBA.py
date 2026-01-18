import collections
import io   # For io.BytesIO
import itertools
import math
import operator
import re
import struct
import sys
import warnings
import zlib
from array import array
fromarray = from_array
def asRGBA(self):
    """
        Return image as RGBA pixels.
        Greyscales are expanded into RGB triplets;
        an alpha channel is synthesized if necessary.
        The return values are as for the :meth:`read` method except that
        the *info* reflect the returned pixels, not the source image.
        In particular, for this method
        ``info['greyscale']`` will be ``False``, and
        ``info['alpha']`` will be ``True``.
        """
    width, height, pixels, info = self.asDirect()
    if info['alpha'] and (not info['greyscale']):
        return (width, height, pixels, info)
    typecode = 'BH'[info['bitdepth'] > 8]
    maxval = 2 ** info['bitdepth'] - 1
    maxbuffer = struct.pack('=' + typecode, maxval) * 4 * width
    if info['bitdepth'] > 8:

        def newarray():
            return array('H', maxbuffer)
    else:

        def newarray():
            return bytearray(maxbuffer)
    if info['alpha'] and info['greyscale']:

        def convert():
            for row in pixels:
                a = newarray()
                convert_la_to_rgba(row, a)
                yield a
    elif info['greyscale']:

        def convert():
            for row in pixels:
                a = newarray()
                convert_l_to_rgba(row, a)
                yield a
    else:
        assert not info['alpha'] and (not info['greyscale'])

        def convert():
            for row in pixels:
                a = newarray()
                convert_rgb_to_rgba(row, a)
                yield a
    info['alpha'] = True
    info['greyscale'] = False
    info['planes'] = 4
    return (width, height, convert(), info)