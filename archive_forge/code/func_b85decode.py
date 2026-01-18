import re
import struct
import binascii
def b85decode(b):
    """Decode the base85-encoded bytes-like object or ASCII string b

    The result is returned as a bytes object.
    """
    global _b85dec
    if _b85dec is None:
        _b85dec = [None] * 256
        for i, c in enumerate(_b85alphabet):
            _b85dec[c] = i
    b = _bytes_from_decode_data(b)
    padding = -len(b) % 5
    b = b + b'~' * padding
    out = []
    packI = struct.Struct('!I').pack
    for i in range(0, len(b), 5):
        chunk = b[i:i + 5]
        acc = 0
        try:
            for c in chunk:
                acc = acc * 85 + _b85dec[c]
        except TypeError:
            for j, c in enumerate(chunk):
                if _b85dec[c] is None:
                    raise ValueError('bad base85 character at position %d' % (i + j)) from None
            raise
        try:
            out.append(packI(acc))
        except struct.error:
            raise ValueError('base85 overflow in hunk starting at byte %d' % i) from None
    result = b''.join(out)
    if padding:
        result = result[:-padding]
    return result