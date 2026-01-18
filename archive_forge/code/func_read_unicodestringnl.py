import codecs
import io
import pickle
import re
import sys
from struct import unpack as _unpack
from pickle import decode_long
def read_unicodestringnl(f):
    """
    >>> import io
    >>> read_unicodestringnl(io.BytesIO(b"abc\\\\uabcd\\njunk")) == 'abc\\uabcd'
    True
    """
    data = f.readline()
    if not data.endswith(b'\n'):
        raise ValueError('no newline found when trying to read unicodestringnl')
    data = data[:-1]
    return str(data, 'raw-unicode-escape')