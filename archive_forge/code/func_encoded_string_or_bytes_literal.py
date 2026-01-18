from __future__ import absolute_import
import re
import sys
def encoded_string_or_bytes_literal(s, encoding):
    if isinstance(s, bytes):
        return bytes_literal(s, encoding)
    else:
        return encoded_string(s, encoding)