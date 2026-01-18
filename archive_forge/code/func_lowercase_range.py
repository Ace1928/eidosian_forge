from __future__ import absolute_import
import types
from . import Errors
def lowercase_range(code1, code2):
    """
    If the range of characters from code1 to code2-1 includes any
    upper case letters, return the corresponding lower case range.
    """
    code3 = max(code1, ord('A'))
    code4 = min(code2, ord('Z') + 1)
    if code3 < code4:
        d = ord('a') - ord('A')
        return (code3 + d, code4 + d)
    else:
        return None