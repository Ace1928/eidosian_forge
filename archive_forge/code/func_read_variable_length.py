from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
def read_variable_length(data):
    """
    Read a variable length variable from the given data.

    Parameters
    ----------
    data : bytearray
        Data of variable length.

    Returns
    -------
    length : int
        Length in bytes.

    """
    next_byte = 1
    value = 0
    while next_byte:
        next_value = byte2int(next(data))
        if not next_value & 128:
            next_byte = 0
        next_value &= 127
        value <<= 7
        value += next_value
    return value