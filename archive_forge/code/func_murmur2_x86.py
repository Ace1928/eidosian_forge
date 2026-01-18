import os
import sys
import tempfile
from IPython.core.compilerop import CachingCompiler
def murmur2_x86(data, seed):
    """Get the murmur2 hash."""
    m = 1540483477
    data = [chr(d) for d in str.encode(data, 'utf8')]
    length = len(data)
    h = seed ^ length
    rounded_end = length & 4294967292
    for i in range(0, rounded_end, 4):
        k = ord(data[i]) & 255 | (ord(data[i + 1]) & 255) << 8 | (ord(data[i + 2]) & 255) << 16 | ord(data[i + 3]) << 24
        k = k * m & 4294967295
        k ^= k >> 24
        k = k * m & 4294967295
        h = h * m & 4294967295
        h ^= k
    val = length & 3
    k = 0
    if val == 3:
        k = (ord(data[rounded_end + 2]) & 255) << 16
    if val in [2, 3]:
        k |= (ord(data[rounded_end + 1]) & 255) << 8
    if val in [1, 2, 3]:
        k |= ord(data[rounded_end]) & 255
        h ^= k
        h = h * m & 4294967295
    h ^= h >> 13
    h = h * m & 4294967295
    h ^= h >> 15
    return h