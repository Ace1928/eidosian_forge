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
def check_palette(palette):
    """
    Check a palette argument (to the :class:`Writer` class) for validity.
    Returns the palette as a list if okay;
    raises an exception otherwise.
    """
    if palette is None:
        return None
    p = list(palette)
    if not 0 < len(p) <= 256:
        raise ProtocolError('a palette must have between 1 and 256 entries, see https://www.w3.org/TR/PNG/#11PLTE')
    seen_triple = False
    for i, t in enumerate(p):
        if len(t) not in (3, 4):
            raise ProtocolError(f'palette entry {i}: entries must be 3- or 4-tuples.')
        if len(t) == 3:
            seen_triple = True
        if seen_triple and len(t) == 4:
            raise ProtocolError(f'palette entry {i}: all 4-tuples must precede all 3-tuples')
        for x in t:
            if int(x) != x or not 0 <= x <= 255:
                raise ProtocolError(f'palette entry {i}: values must be integer: 0 <= x <= 255')
    return p