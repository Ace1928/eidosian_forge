import codecs
import io
import pickle
import re
import sys
from struct import unpack as _unpack
from pickle import decode_long
def read_stringnl_noescape_pair(f):
    """
    >>> import io
    >>> read_stringnl_noescape_pair(io.BytesIO(b"Queue\\nEmpty\\njunk"))
    'Queue Empty'
    """
    return '%s %s' % (read_stringnl_noescape(f), read_stringnl_noescape(f))