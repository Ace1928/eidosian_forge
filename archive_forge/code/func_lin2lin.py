import math
import struct
from ctypes import create_string_buffer
def lin2lin(cp, size, size2):
    _check_params(len(cp), size)
    _check_size(size2)
    if size == size2:
        return cp
    new_len = len(cp) / size * size2
    result = create_string_buffer(new_len)
    for i in range(_sample_count(cp, size)):
        sample = _get_sample(cp, size, i)
        if size < size2:
            sample = sample << 4 * size2 / size
        elif size > size2:
            sample = sample >> 4 * size / size2
        sample = _overflow(sample, size2)
        _put_sample(result, size2, i, sample)
    return result.raw