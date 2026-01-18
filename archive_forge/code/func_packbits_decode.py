from __future__ import annotations
from typing import Any, Literal, overload
import numpy
def packbits_decode(encoded: bytes, /, *, out=None) -> bytes:
    """Decompress PackBits encoded byte string.

    >>> packbits_decode(b'\\x80\\x80')  # NOP
    b''
    >>> packbits_decode(b'\\x02123')
    b'123'
    >>> packbits_decode(
    ...   b'\\xfe\\xaa\\x02\\x80\\x00\\x2a\\xfd\\xaa\\x03\\x80\\x00\\x2a\\x22\\xf7\\xaa'
    ...     )[:-5]
    b'\\xaa\\xaa\\xaa\\x80\\x00*\\xaa\\xaa\\xaa\\xaa\\x80\\x00*"\\xaa\\xaa\\xaa\\xaa\\xaa'

    """
    out = []
    out_extend = out.extend
    i = 0
    try:
        while True:
            n = ord(encoded[i:i + 1]) + 1
            i += 1
            if n > 129:
                out_extend(encoded[i:i + 1] * (258 - n))
                i += 1
            elif n < 129:
                out_extend(encoded[i:i + n])
                i += n
    except TypeError:
        pass
    return bytes(out)