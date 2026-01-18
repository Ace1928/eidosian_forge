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
def iteridat():
    """Iterator that yields all the ``IDAT`` chunks as strings."""
    while True:
        type, data = self.chunk(lenient=lenient)
        if type == b'IEND':
            break
        if type != b'IDAT':
            continue
        if self.colormap and (not self.plte):
            warnings.warn('PLTE chunk is required before IDAT chunk')
        yield data