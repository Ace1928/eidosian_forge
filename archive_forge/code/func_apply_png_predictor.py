import io
import pathlib
import string
import struct
from html import escape
from typing import (
import charset_normalizer  # For str encoding detection
def apply_png_predictor(pred: int, colors: int, columns: int, bitspercomponent: int, data: bytes) -> bytes:
    """Reverse the effect of the PNG predictor

    Documentation: http://www.libpng.org/pub/png/spec/1.2/PNG-Filters.html
    """
    if bitspercomponent not in [8, 1]:
        msg = "Unsupported `bitspercomponent': %d" % bitspercomponent
        raise ValueError(msg)
    nbytes = colors * columns * bitspercomponent // 8
    bpp = colors * bitspercomponent // 8
    buf = b''
    line_above = b'\x00' * columns
    for scanline_i in range(0, len(data), nbytes + 1):
        filter_type = data[scanline_i]
        line_encoded = data[scanline_i + 1:scanline_i + 1 + nbytes]
        raw = b''
        if filter_type == 0:
            raw += line_encoded
        elif filter_type == 1:
            for j, sub_x in enumerate(line_encoded):
                if j - bpp < 0:
                    raw_x_bpp = 0
                else:
                    raw_x_bpp = int(raw[j - bpp])
                raw_x = sub_x + raw_x_bpp & 255
                raw += bytes((raw_x,))
        elif filter_type == 2:
            for up_x, prior_x in zip(line_encoded, line_above):
                raw_x = up_x + prior_x & 255
                raw += bytes((raw_x,))
        elif filter_type == 3:
            for j, average_x in enumerate(line_encoded):
                if j - bpp < 0:
                    raw_x_bpp = 0
                else:
                    raw_x_bpp = int(raw[j - bpp])
                prior_x = int(line_above[j])
                raw_x = average_x + (raw_x_bpp + prior_x) // 2 & 255
                raw += bytes((raw_x,))
        elif filter_type == 4:
            for j, paeth_x in enumerate(line_encoded):
                if j - bpp < 0:
                    raw_x_bpp = 0
                    prior_x_bpp = 0
                else:
                    raw_x_bpp = int(raw[j - bpp])
                    prior_x_bpp = int(line_above[j - bpp])
                prior_x = int(line_above[j])
                paeth = paeth_predictor(raw_x_bpp, prior_x, prior_x_bpp)
                raw_x = paeth_x + paeth & 255
                raw += bytes((raw_x,))
        else:
            raise ValueError('Unsupported predictor value: %d' % filter_type)
        buf += raw
        line_above = raw
    return buf