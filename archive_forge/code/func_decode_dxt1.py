from __future__ import annotations
import os
import struct
from enum import IntEnum
from io import BytesIO
from . import Image, ImageFile
def decode_dxt1(data, alpha=False):
    """
    input: one "row" of data (i.e. will produce 4*width pixels)
    """
    blocks = len(data) // 8
    ret = (bytearray(), bytearray(), bytearray(), bytearray())
    for block in range(blocks):
        idx = block * 8
        color0, color1, bits = struct.unpack_from('<HHI', data, idx)
        r0, g0, b0 = unpack_565(color0)
        r1, g1, b1 = unpack_565(color1)
        for j in range(4):
            for i in range(4):
                control = bits & 3
                bits = bits >> 2
                a = 255
                if control == 0:
                    r, g, b = (r0, g0, b0)
                elif control == 1:
                    r, g, b = (r1, g1, b1)
                elif control == 2:
                    if color0 > color1:
                        r = (2 * r0 + r1) // 3
                        g = (2 * g0 + g1) // 3
                        b = (2 * b0 + b1) // 3
                    else:
                        r = (r0 + r1) // 2
                        g = (g0 + g1) // 2
                        b = (b0 + b1) // 2
                elif control == 3:
                    if color0 > color1:
                        r = (2 * r1 + r0) // 3
                        g = (2 * g1 + g0) // 3
                        b = (2 * b1 + b0) // 3
                    else:
                        r, g, b, a = (0, 0, 0, 0)
                if alpha:
                    ret[j].extend([r, g, b, a])
                else:
                    ret[j].extend([r, g, b])
    return ret