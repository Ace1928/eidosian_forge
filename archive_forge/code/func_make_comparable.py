import base64
import sys
import time
from datetime import datetime
from decimal import Decimal
import http.client
import urllib.parse
from xml.parsers import expat
import errno
from io import BytesIO
def make_comparable(self, other):
    if isinstance(other, DateTime):
        s = self.value
        o = other.value
    elif isinstance(other, datetime):
        s = self.value
        o = _iso8601_format(other)
    elif isinstance(other, str):
        s = self.value
        o = other
    elif hasattr(other, 'timetuple'):
        s = self.timetuple()
        o = other.timetuple()
    else:
        s = self
        o = NotImplemented
    return (s, o)