from __future__ import division
import re
import stat
from .helpers import (
def format_property(name, value):
    """Format the name and value (both unicode) of a property as a string."""
    result = b''
    utf8_name = utf8_bytes_string(name)
    result = b'property ' + utf8_name
    if value is not None:
        utf8_value = utf8_bytes_string(value)
        result += b' ' + ('%d' % len(utf8_value)).encode('ascii') + b' ' + utf8_value
    return result