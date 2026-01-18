from __future__ import with_statement
import inspect
import keyword
import os
import re
import sys
import time
import tokenize
import warnings
from fnmatch import fnmatch
from optparse import OptionParser
def extraneous_whitespace(logical_line):
    """Avoid extraneous whitespace.

    Avoid extraneous whitespace in these situations:
    - Immediately inside parentheses, brackets or braces.
    - Immediately before a comma, semicolon, or colon.

    Okay: spam(ham[1], {eggs: 2})
    E201: spam( ham[1], {eggs: 2})
    E201: spam(ham[ 1], {eggs: 2})
    E201: spam(ham[1], { eggs: 2})
    E202: spam(ham[1], {eggs: 2} )
    E202: spam(ham[1 ], {eggs: 2})
    E202: spam(ham[1], {eggs: 2 })

    E203: if x == 4: print(x, y); x, y = y , x
    E203: if x == 4: print(x, y); x, y = y, x
    E203: if x == 4 : print(x, y); x, y = y, x
    """
    line = logical_line
    for match in EXTRANEOUS_WHITESPACE_REGEX.finditer(line):
        text = match.group()
        char = text.strip()
        found = match.start()
        if text == char + ' ':
            yield (found + 1, "E201 whitespace after '%s'" % char)
        elif line[found - 1] != ',':
            code = 'E202' if char in '}])' else 'E203'
            yield (found, "%s whitespace before '%s'" % (code, char))