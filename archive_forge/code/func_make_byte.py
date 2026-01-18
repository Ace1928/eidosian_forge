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
def make_byte(block):
    """Take a block of (2, 4, or 8) values,
        and pack them into a single byte.
        """
    res = 0
    for v in block:
        res = (res << bitdepth) + v
    return res