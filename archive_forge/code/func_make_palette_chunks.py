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
def make_palette_chunks(palette):
    """
    Create the byte sequences for a ``PLTE`` and
    if necessary a ``tRNS`` chunk.
    Returned as a pair (*p*, *t*).
    *t* will be ``None`` if no ``tRNS`` chunk is necessary.
    """
    p = bytearray()
    t = bytearray()
    for x in palette:
        p.extend(x[0:3])
        if len(x) > 3:
            t.append(x[3])
    if t:
        return (p, t)
    return (p, None)