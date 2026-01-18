from __future__ import absolute_import
import re
import sys
def encode_pyunicode_string(s):
    """Create Py_UNICODE[] representation of a given unicode string.
    """
    s = list(map(ord, s)) + [0]
    if sys.maxunicode >= 65536:
        utf16, utf32 = ([], s)
        for code_point in s:
            if code_point >= 65536:
                high, low = divmod(code_point - 65536, 1024)
                utf16.append(high + 55296)
                utf16.append(low + 56320)
            else:
                utf16.append(code_point)
    else:
        utf16, utf32 = (s, [])
        for code_unit in s:
            if 56320 <= code_unit <= 57343 and utf32 and (55296 <= utf32[-1] <= 56319):
                high, low = (utf32[-1], code_unit)
                utf32[-1] = ((high & 1023) << 10) + (low & 1023) + 65536
            else:
                utf32.append(code_unit)
    if utf16 == utf32:
        utf16 = []
    return (','.join(map(_unicode, utf16)), ','.join(map(_unicode, utf32)))