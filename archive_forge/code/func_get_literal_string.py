from __future__ import annotations
import calendar
import codecs
import collections
import mmap
import os
import re
import time
import zlib
@classmethod
def get_literal_string(cls, data, offset):
    nesting_depth = 0
    result = bytearray()
    for m in cls.re_lit_str_token.finditer(data, offset):
        result.extend(data[offset:m.start()])
        if m.group(1):
            result.extend(cls.escaped_chars[m.group(1)[1]])
        elif m.group(2):
            result.append(int(m.group(2)[1:], 8))
        elif m.group(3):
            pass
        elif m.group(5):
            result.extend(b'\n')
        elif m.group(6):
            result.extend(b'(')
            nesting_depth += 1
        elif m.group(7):
            if nesting_depth == 0:
                return (bytes(result), m.end())
            result.extend(b')')
            nesting_depth -= 1
        offset = m.end()
    msg = 'unfinished literal string'
    raise PdfFormatError(msg)