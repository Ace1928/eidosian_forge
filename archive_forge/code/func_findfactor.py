import math
import struct
from ctypes import create_string_buffer
def findfactor(cp1, cp2):
    size = 2
    if len(cp1) % 2 != 0:
        raise error('Strings should be even-sized')
    if len(cp1) != len(cp2):
        raise error('Samples should be same size')
    sample_count = _sample_count(cp1, size)
    sum_ri_2 = _sum2(cp2, cp2, sample_count)
    sum_aij_ri = _sum2(cp1, cp2, sample_count)
    return sum_aij_ri / sum_ri_2