from unittest import TestCase
from simplejson.compat import StringIO, long_type, b, binary_type, text_type, PY3
import simplejson as json
def as_text_type(s):
    if PY3 and isinstance(s, bytes):
        return s.decode('ascii')
    return s