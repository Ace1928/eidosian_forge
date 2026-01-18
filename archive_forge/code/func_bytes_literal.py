from __future__ import absolute_import
import re
import sys
def bytes_literal(s, encoding):
    assert isinstance(s, bytes)
    s = BytesLiteral(s)
    s.encoding = encoding
    return s