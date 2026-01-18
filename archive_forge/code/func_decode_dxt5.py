from __future__ import annotations
import os
import struct
from enum import IntEnum
from io import BytesIO
from . import Image, ImageFile
def decode_dxt5(data):
    """
    input: one "row" of data (i.e. will produce 4 * width pixels)
    """
    blocks = len(data) // 16
    ret = (bytearray(), bytearray(), bytearray(), bytearray())
    for block in range(blocks):
        idx = block * 16
        block = data[idx:idx + 16]
        a0, a1 = struct.unpack_from('<BB', block)
        bits = struct.unpack_from('<6B', block, 2)
        alphacode1 = bits[2] | bits[3] << 8 | bits[4] << 16 | bits[5] << 24
        alphacode2 = bits[0] | bits[1] << 8
        color0, color1 = struct.unpack_from('<HH', block, 8)
        code, = struct.unpack_from('<I', block, 12)
        r0, g0, b0 = unpack_565(color0)
        r1, g1, b1 = unpack_565(color1)
        for j in range(4):
            for i in range(4):
                alphacode_index = 3 * (4 * j + i)
                if alphacode_index <= 12:
                    alphacode = alphacode2 >> alphacode_index & 7
                elif alphacode_index == 15:
                    alphacode = alphacode2 >> 15 | alphacode1 << 1 & 6
                else:
                    alphacode = alphacode1 >> alphacode_index - 16 & 7
                if alphacode == 0:
                    a = a0
                elif alphacode == 1:
                    a = a1
                elif a0 > a1:
                    a = ((8 - alphacode) * a0 + (alphacode - 1) * a1) // 7
                elif alphacode == 6:
                    a = 0
                elif alphacode == 7:
                    a = 255
                else:
                    a = ((6 - alphacode) * a0 + (alphacode - 1) * a1) // 5
                color_code = code >> 2 * (4 * j + i) & 3
                if color_code == 0:
                    r, g, b = (r0, g0, b0)
                elif color_code == 1:
                    r, g, b = (r1, g1, b1)
                elif color_code == 2:
                    r = (2 * r0 + r1) // 3
                    g = (2 * g0 + g1) // 3
                    b = (2 * b0 + b1) // 3
                elif color_code == 3:
                    r = (2 * r1 + r0) // 3
                    g = (2 * g1 + g0) // 3
                    b = (2 * b1 + b0) // 3
                ret[j].extend([r, g, b, a])
    return ret