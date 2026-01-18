import os
import zlib
import time  # noqa
import logging
import numpy as np
def get_type_and_len(bb):
    """bb should be 6 bytes at least
    Return (type, length, length_of_full_tag)
    """
    value = ''
    for i in range(2):
        b = bb[i:i + 1]
        tmp = bin(ord(b))[2:]
        value = tmp.rjust(8, '0') + value
    type = int(value[:10], 2)
    L = int(value[10:], 2)
    L2 = L + 2
    if L == 63:
        value = ''
        for i in range(2, 6):
            b = bb[i:i + 1]
            tmp = bin(ord(b))[2:]
            value = tmp.rjust(8, '0') + value
        L = int(value, 2)
        L2 = L + 6
    return (type, L, L2)