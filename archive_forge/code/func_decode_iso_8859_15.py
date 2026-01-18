from unittest import TestCase
from simplejson.compat import StringIO, long_type, b, binary_type, text_type, PY3
import simplejson as json
def decode_iso_8859_15(b):
    return b.decode('iso-8859-15')