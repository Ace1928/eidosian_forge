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
def asRGB(self):
    """
        Return image as RGB pixels.
        RGB colour images are passed through unchanged;
        greyscales are expanded into RGB triplets
        (there is a small speed overhead for doing this).

        An alpha channel in the source image will raise an exception.

        The return values are as for the :meth:`read` method except that
        the *info* reflect the returned pixels, not the source image.
        In particular,
        for this method ``info['greyscale']`` will be ``False``.
        """
    width, height, pixels, info = self.asDirect()
    if info['alpha']:
        raise Error('will not convert image with alpha channel to RGB')
    if not info['greyscale']:
        return (width, height, pixels, info)
    info['greyscale'] = False
    info['planes'] = 3
    if info['bitdepth'] > 8:

        def newarray():
            return array('H', [0])
    else:

        def newarray():
            return bytearray([0])

    def iterrgb():
        for row in pixels:
            a = newarray() * 3 * width
            for i in range(3):
                a[i::3] = row
            yield a
    return (width, height, iterrgb(), info)