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
def check_bitdepth_colortype(bitdepth, colortype):
    """
    Check that `bitdepth` and `colortype` are both valid,
    and specified in a valid combination.
    Returns (None) if valid, raise an Exception if not valid.
    """
    if bitdepth not in (1, 2, 4, 8, 16):
        raise FormatError(f'invalid bit depth {bitdepth}')
    if colortype not in (0, 2, 3, 4, 6):
        raise FormatError(f'invalid colour type {colortype}')
    if colortype & 1 and bitdepth > 8:
        raise FormatError(f'Indexed images (colour type {bitdepth}) cannot have bitdepth > 8 (bit depth {colortype}). See http://www.w3.org/TR/2003/REC-PNG-20031110/#table111 .')
    if bitdepth < 8 and colortype not in (0, 3):
        raise FormatError(f'Illegal combination of bit depth ({bitdepth}) and colour type ({colortype}). See http://www.w3.org/TR/2003/REC-PNG-20031110/#table111 .')