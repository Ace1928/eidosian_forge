import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def oneOf(strs, caseless=False, useRegex=True):
    """Helper to quickly define a set of alternative Literals, and makes sure to do
       longest-first testing when there is a conflict, regardless of the input order,
       but returns a C{L{MatchFirst}} for best performance.

       Parameters:
        - strs - a string of space-delimited literals, or a list of string literals
        - caseless - (default=False) - treat all literals as caseless
        - useRegex - (default=True) - as an optimization, will generate a Regex
          object; otherwise, will generate a C{MatchFirst} object (if C{caseless=True}, or
          if creating a C{Regex} raises an exception)
    """
    if caseless:
        isequal = lambda a, b: a.upper() == b.upper()
        masks = lambda a, b: b.upper().startswith(a.upper())
        parseElementClass = CaselessLiteral
    else:
        isequal = lambda a, b: a == b
        masks = lambda a, b: b.startswith(a)
        parseElementClass = Literal
    if isinstance(strs, (list, tuple)):
        symbols = list(strs[:])
    elif isinstance(strs, basestring):
        symbols = strs.split()
    else:
        warnings.warn('Invalid argument to oneOf, expected string or list', SyntaxWarning, stacklevel=2)
    i = 0
    while i < len(symbols) - 1:
        cur = symbols[i]
        for j, other in enumerate(symbols[i + 1:]):
            if isequal(other, cur):
                del symbols[i + j + 1]
                break
            elif masks(cur, other):
                del symbols[i + j + 1]
                symbols.insert(i, other)
                cur = other
                break
        else:
            i += 1
    if not caseless and useRegex:
        try:
            if len(symbols) == len(''.join(symbols)):
                return Regex('[%s]' % ''.join((_escapeRegexRangeChars(sym) for sym in symbols)))
            else:
                return Regex('|'.join((re.escape(sym) for sym in symbols)))
        except:
            warnings.warn('Exception creating Regex for oneOf, building MatchFirst', SyntaxWarning, stacklevel=2)
    return MatchFirst([parseElementClass(sym) for sym in symbols])