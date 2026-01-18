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
def explicit_line_join(logical_line, tokens):
    """Avoid explicit line join between brackets.

    The preferred way of wrapping long lines is by using Python's implied line
    continuation inside parentheses, brackets and braces.  Long lines can be
    broken over multiple lines by wrapping expressions in parentheses.  These
    should be used in preference to using a backslash for line continuation.

    E502: aaa = [123, \\\\n       123]
    E502: aaa = ("bbb " \\\\n       "ccc")

    Okay: aaa = [123,\\n       123]
    Okay: aaa = ("bbb "\\n       "ccc")
    Okay: aaa = "bbb " \\\\n    "ccc"
    Okay: aaa = 123  # \\\\
    """
    prev_start = prev_end = parens = 0
    comment = False
    backslash = None
    for token_type, text, start, end, line in tokens:
        if token_type == tokenize.COMMENT:
            comment = True
        if start[0] != prev_start and parens and backslash and (not comment):
            yield (backslash, 'E502 the backslash is redundant between brackets')
        if end[0] != prev_end:
            if line.rstrip('\r\n').endswith('\\'):
                backslash = (end[0], len(line.splitlines()[-1]) - 1)
            else:
                backslash = None
            prev_start = prev_end = end[0]
        else:
            prev_start = start[0]
        if token_type == tokenize.OP:
            if text in '([{':
                parens += 1
            elif text in ')]}':
                parens -= 1