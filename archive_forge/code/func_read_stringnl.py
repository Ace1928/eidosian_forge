import codecs
import io
import pickle
import re
import sys
from struct import unpack as _unpack
from pickle import decode_long
def read_stringnl(f, decode=True, stripquotes=True):
    """
    >>> import io
    >>> read_stringnl(io.BytesIO(b"'abcd'\\nefg\\n"))
    'abcd'

    >>> read_stringnl(io.BytesIO(b"\\n"))
    Traceback (most recent call last):
    ...
    ValueError: no string quotes around b''

    >>> read_stringnl(io.BytesIO(b"\\n"), stripquotes=False)
    ''

    >>> read_stringnl(io.BytesIO(b"''\\n"))
    ''

    >>> read_stringnl(io.BytesIO(b'"abcd"'))
    Traceback (most recent call last):
    ...
    ValueError: no newline found when trying to read stringnl

    Embedded escapes are undone in the result.
    >>> read_stringnl(io.BytesIO(br"'a\\n\\\\b\\x00c\\td'" + b"\\n'e'"))
    'a\\n\\\\b\\x00c\\td'
    """
    data = f.readline()
    if not data.endswith(b'\n'):
        raise ValueError('no newline found when trying to read stringnl')
    data = data[:-1]
    if stripquotes:
        for q in (b'"', b"'"):
            if data.startswith(q):
                if not data.endswith(q):
                    raise ValueError('strinq quote %r not found at both ends of %r' % (q, data))
                data = data[1:-1]
                break
        else:
            raise ValueError('no string quotes around %r' % data)
    if decode:
        data = codecs.escape_decode(data)[0].decode('ascii')
    return data