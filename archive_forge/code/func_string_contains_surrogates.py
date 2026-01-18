from __future__ import absolute_import
import re
import sys
def string_contains_surrogates(ustring):
    """
    Check if the unicode string contains surrogate code points
    on a CPython platform with wide (UCS-4) or narrow (UTF-16)
    Unicode, i.e. characters that would be spelled as two
    separate code units on a narrow platform.
    """
    for c in map(ord, ustring):
        if c > 65535:
            return True
        if 55296 <= c <= 57343:
            return True
    return False