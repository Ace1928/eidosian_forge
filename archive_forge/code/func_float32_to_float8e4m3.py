import collections.abc
import numbers
import struct
from cmath import isnan
from typing import (
import google.protobuf.message
import numpy as np
from onnx import (
def float32_to_float8e4m3(fval: float, scale: float=1.0, fn: bool=True, uz: bool=False, saturate: bool=True) -> int:
    """Convert a float32 value to a float8, e4m3 (as int).

    See :ref:`onnx-detail-float8` for technical details.

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
    if not fn:
        raise NotImplementedError('float32_to_float8e4m3 not implemented with fn=False.')
    x = fval / scale
    b = int.from_bytes(struct.pack('<f', np.float32(x)), 'little')
    ret = (b & 2147483648) >> 24
    if uz:
        if b & 2143289344 == 2143289344:
            return 128
        if np.isinf(x):
            if saturate:
                return ret | 127
            return 128
        e = (b & 2139095040) >> 23
        m = b & 8388607
        if e < 116:
            ret = 0
        elif e < 120:
            ex = e - 119
            if ex >= -2:
                ret |= 1 << 2 + ex
                ret |= m >> 21 - ex
            elif m > 0:
                ret |= 1
            else:
                ret = 0
            mask = 1 << 20 - ex
            if m & mask and (ret & 1 or m & mask - 1 > 0 or (m & mask and m & mask << 1 and (m & mask - 1 == 0))):
                ret += 1
        elif e < 135:
            ex = e - 119
            if ex == 0:
                ret |= 4
                ret |= m >> 21
            else:
                ret |= ex << 3
                ret |= m >> 20
            if m & 524288 and (m & 1048576 or m & 524287):
                if ret & 127 < 127:
                    ret += 1
                elif not saturate:
                    return 128
        elif saturate:
            ret |= 127
        else:
            ret = 128
        return int(ret)
    else:
        if b & 2143289344 == 2143289344:
            return 127 | ret
        if np.isinf(x):
            if saturate:
                return ret | 126
            return 127 | ret
        e = (b & 2139095040) >> 23
        m = b & 8388607
        if e != 0:
            if e < 117:
                pass
            elif e < 121:
                ex = e - 120
                if ex >= -2:
                    ret |= 1 << 2 + ex
                    ret |= m >> 21 - ex
                elif m > 0:
                    ret |= 1
                mask = 1 << 20 - ex
                if m & mask and (ret & 1 or m & mask - 1 > 0 or (m & mask and m & mask << 1 and (m & mask - 1 == 0))):
                    ret += 1
            elif e < 136:
                ex = e - 120
                if ex == 0:
                    ret |= 4
                    ret |= m >> 21
                else:
                    ret |= ex << 3
                    ret |= m >> 20
                    if ret & 127 == 127:
                        ret &= 254
                if m & 524288 and (m & 1048576 or m & 524287):
                    if ret & 127 < 126:
                        ret += 1
                    elif not saturate:
                        ret |= 127
            elif saturate:
                ret |= 126
            else:
                ret |= 127
        return int(ret)