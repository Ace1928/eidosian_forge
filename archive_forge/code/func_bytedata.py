import struct
import sys
import numpy as np
def bytedata(a):
    """Return a the raw data corresponding to a."""
    if isinstance(a, (bytearray, bytes, memoryview)):
        return a
    elif hasattr(a, 'data'):
        return a.data
    else:
        raise ValueError(a, 'cannot return bytedata')