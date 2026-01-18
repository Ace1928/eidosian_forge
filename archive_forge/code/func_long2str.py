import math
import sys
import struct
from Cryptodome import Random
from Cryptodome.Util.py3compat import iter_range
import struct
import warnings
def long2str(n, blocksize=0):
    warnings.warn('long2str() has been replaced by long_to_bytes()')
    return long_to_bytes(n, blocksize)