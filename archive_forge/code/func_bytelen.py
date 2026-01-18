import struct
import sys
import numpy as np
def bytelen(a):
    """Determine the length of a in bytes."""
    if hasattr(a, 'nbytes'):
        return a.nbytes
    elif isinstance(a, (bytearray, bytes)):
        return len(a)
    else:
        raise ValueError(a, 'cannot determine nbytes')