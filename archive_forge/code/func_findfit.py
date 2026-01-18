import math
import struct
from ctypes import create_string_buffer
def findfit(cp1, cp2):
    size = 2
    if len(cp1) % 2 != 0 or len(cp2) % 2 != 0:
        raise error('Strings should be even-sized')
    if len(cp1) < len(cp2):
        raise error('First sample should be longer')
    len1 = _sample_count(cp1, size)
    len2 = _sample_count(cp2, size)
    sum_ri_2 = _sum2(cp2, cp2, len2)
    sum_aij_2 = _sum2(cp1, cp1, len2)
    sum_aij_ri = _sum2(cp1, cp2, len2)
    result = (sum_ri_2 * sum_aij_2 - sum_aij_ri * sum_aij_ri) / sum_aij_2
    best_result = result
    best_i = 0
    for i in range(1, len1 - len2 + 1):
        aj_m1 = _get_sample(cp1, size, i - 1)
        aj_lm1 = _get_sample(cp1, size, i + len2 - 1)
        sum_aij_2 += aj_lm1 ** 2 - aj_m1 ** 2
        sum_aij_ri = _sum2(buffer(cp1)[i * size:], cp2, len2)
        result = (sum_ri_2 * sum_aij_2 - sum_aij_ri * sum_aij_ri) / sum_aij_2
        if result < best_result:
            best_result = result
            best_i = i
    factor = _sum2(buffer(cp1)[best_i * size:], cp2, len2) / sum_ri_2
    return (best_i, factor)