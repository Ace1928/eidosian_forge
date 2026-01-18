import sys
def as_8_bit(x, encoding='utf-8'):
    if isinstance(x, unicode):
        return x.encode(encoding)
    elif isinstance(x, bytes):
        return x
    return str(x).encode(encoding)