import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def srange(s):
    """Helper to easily define string ranges for use in Word construction.  Borrows
       syntax from regexp '[]' string range definitions::
          srange("[0-9]")   -> "0123456789"
          srange("[a-z]")   -> "abcdefghijklmnopqrstuvwxyz"
          srange("[a-z$_]") -> "abcdefghijklmnopqrstuvwxyz$_"
       The input string must be enclosed in []'s, and the returned string is the expanded
       character set joined into a single string.
       The values enclosed in the []'s may be::
          a single character
          an escaped character with a leading backslash (such as \\- or \\])
          an escaped hex character with a leading '\\x' (\\x21, which is a '!' character) 
            (\\0x## is also supported for backwards compatibility) 
          an escaped octal character with a leading '\\0' (\\041, which is a '!' character)
          a range of any of the above, separated by a dash ('a-z', etc.)
          any combination of the above ('aeiouy', 'a-zA-Z0-9_$', etc.)
    """
    try:
        return ''.join((_expanded(part) for part in _reBracketExpr.parseString(s).body))
    except:
        return ''