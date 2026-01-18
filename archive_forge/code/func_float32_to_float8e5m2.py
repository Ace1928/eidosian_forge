import collections.abc
import numbers
import struct
from cmath import isnan
from typing import (
import google.protobuf.message
import numpy as np
from onnx import (
def float32_to_float8e5m2(fval: float, scale: float=1.0, fn: bool=False, uz: bool=False, saturate: bool=True) -> int:
    """Convert a float32 value to a float8, e5m2 (as int).

    Args:
        fval: float to convert
        scale: scale, divide *fval* by *scale* before casting it
        fn: no infinite values
        uz: no negative zero
        saturate: if True, any value out of range included inf becomes
            the maximum value, otherwise, it becomes NaN. The
            description of operator Cast fully describes the
            differences.

    Returns:
        converted float
    """
    x = fval / scale
    b = int.from_bytes(struct.pack('<f', np.float32(x)), 'little')
    ret = (b & 2147483648) >> 24
    if fn and uz:
        if b & 2143289344 == 2143289344:
            return 128
        if b & 2147483647 == 2139095040:
            if saturate:
                return ret | 127
            return 128
        e = (b & 2139095040) >> 23
        m = b & 8388607
        if e < 109:
            ret = 0
        elif e < 112:
            ex = e - 111
            if ex >= -1:
                ret |= 1 << 1 + ex
                ret |= m >> 22 - ex
            elif m > 0:
                ret |= 1
            else:
                ret = 0
            mask = 1 << 21 - ex
            if m & mask and (ret & 1 or m & mask - 1 > 0 or (m & mask and m & mask << 1 and (m & mask - 1 == 0))):
                ret += 1
        elif e < 143:
            ex = e - 111
            ret |= ex << 2
            ret |= m >> 21
            if m & 1048576 and (m & 1048575 or m & 2097152):
                if ret & 127 < 127:
                    ret += 1
                elif not saturate:
                    ret = 128
        elif e == 255 and m == 0:
            ret = 128
        elif saturate:
            ret |= 127
        else:
            ret = 128
        return int(ret)
    elif not fn and (not uz):
        if b & 2143289344 == 2143289344:
            return 127 | ret
        if np.isinf(x):
            if saturate:
                return 123 | ret
            return 124 | ret
        e = (b & 2139095040) >> 23
        m = b & 8388607
        if e != 0:
            if e < 110:
                pass
            elif e < 113:
                ex = e - 112
                if ex >= -1:
                    ret |= 1 << 1 + ex
                    ret |= m >> 22 - ex
                elif m > 0:
                    ret |= 1
                mask = 1 << 21 - ex
                if m & mask and (ret & 1 or m & mask - 1 > 0 or (m & mask and m & mask << 1 and (m & mask - 1 == 0))):
                    ret += 1
            elif e < 143:
                ex = e - 112
                ret |= ex << 2
                ret |= m >> 21
                if m & 1048576 and (m & 1048575 or m & 2097152):
                    if ret & 127 < 123:
                        ret += 1
                    elif saturate:
                        ret |= 123
                    else:
                        ret |= 124
            elif saturate:
                ret |= 123
            else:
                ret |= 124
        return int(ret)
    else:
        raise NotImplementedError('fn and uz must be both False or True.')