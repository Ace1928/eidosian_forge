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
def check_color(c, greyscale, which):
    """
    Checks that a colour argument for transparent or background options
    is the right form.
    Returns the colour
    (which, if it's a bare integer, is "corrected" to a 1-tuple).
    """
    if c is None:
        return c
    if greyscale:
        try:
            len(c)
        except TypeError:
            c = (c,)
        if len(c) != 1:
            raise ProtocolError(f'{which} for greyscale must be 1-tuple')
        if not is_natural(c[0]):
            raise ProtocolError(f'{which} colour for greyscale must be integer')
    elif not (len(c) == 3 and is_natural(c[0]) and is_natural(c[1]) and is_natural(c[2])):
        raise ProtocolError(f'{which} colour must be a triple of integers')
    return c