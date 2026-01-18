import struct
import sys
import numpy as np
def check_acceptable_input_type(data, allow64):
    """Check that the data has an acceptable type for tensor encoding.

    :param data: array
    :param allow64: allow 64 bit types
    """
    for a in data:
        if a.dtype.name not in long_to_short:
            raise ValueError('unsupported dataypte')
        if not allow64 and a.dtype.name not in ['float64', 'int64', 'uint64']:
            raise ValueError('64 bit datatypes not allowed unless explicitly enabled')