from __future__ import absolute_import
import re
import sys
def string_contains_lone_surrogates(ustring):
    """
    Check if the unicode string contains lone surrogate code points
    on a CPython platform with wide (UCS-4) or narrow (UTF-16)
    Unicode, i.e. characters that would be spelled as two
    separate code units on a narrow platform, but that do not form a pair.
    """
    last_was_start = False
    unicode_uses_surrogate_encoding = sys.maxunicode == 65535
    for c in map(ord, ustring):
        if c < 55296 or c > 57343:
            if last_was_start:
                return True
        elif not unicode_uses_surrogate_encoding:
            return True
        elif c <= 56319:
            if last_was_start:
                return True
            last_was_start = True
        else:
            if not last_was_start:
                return True
            last_was_start = False
    return last_was_start