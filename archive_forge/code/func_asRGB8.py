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
def asRGB8(self):
    """
        Return the image data as an RGB pixels with 8-bits per sample.
        This is like the :meth:`asRGB` method except that
        this method additionally rescales the values so that
        they are all between 0 and 255 (8-bit).
        In the case where the source image has a bit depth < 8
        the transformation preserves all the information;
        where the source image has bit depth > 8, then
        rescaling to 8-bit values loses precision.
        No dithering is performed.
        Like :meth:`asRGB`,
        an alpha channel in the source image will raise an exception.

        This function returns a 4-tuple:
        (*width*, *height*, *rows*, *info*).
        *width*, *height*, *info* are as per the :meth:`read` method.

        *rows* is the pixel data as a sequence of rows.
        """
    return self._as_rescale(self.asRGB, 8)