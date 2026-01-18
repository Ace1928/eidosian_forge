from __future__ import print_function
import re
from sqlparse import tokens as T
from sqlparse.compat import string_types, text_type, unicode_compatible
from sqlparse.utils import imt, remove_quotes
def get_cases(self, skip_ws=False):
    """Returns a list of 2-tuples (condition, value).

        If an ELSE exists condition is None.
        """
    CONDITION = 1
    VALUE = 2
    ret = []
    mode = CONDITION
    for token in self.tokens:
        if token.match(T.Keyword, 'CASE'):
            continue
        elif skip_ws and token.ttype in T.Whitespace:
            continue
        elif token.match(T.Keyword, 'WHEN'):
            ret.append(([], []))
            mode = CONDITION
        elif token.match(T.Keyword, 'THEN'):
            mode = VALUE
        elif token.match(T.Keyword, 'ELSE'):
            ret.append((None, []))
            mode = VALUE
        elif token.match(T.Keyword, 'END'):
            mode = None
        if mode and (not ret):
            ret.append(([], []))
        if mode == CONDITION:
            ret[-1][0].append(token)
        elif mode == VALUE:
            ret[-1][1].append(token)
    return ret