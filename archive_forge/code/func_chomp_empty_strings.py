import decimal
import json as _json
import sys
import re
from functools import reduce
from _plotly_utils.optional_imports import get_module
from _plotly_utils.basevalidators import ImageUriValidator
def chomp_empty_strings(strings, c, reverse=False):
    """
    Given a list of strings, some of which are the empty string "", replace the
    empty strings with c and combine them with the closest non-empty string on
    the left or "" if it is the first string.
    Examples:
    for c="_"
    ['hey', '', 'why', '', '', 'whoa', '', ''] -> ['hey_', 'why__', 'whoa__']
    ['', 'hi', '', "I'm", 'bob', '', ''] -> ['_', 'hi_', "I'm", 'bob__']
    ['hi', "i'm", 'a', 'good', 'string'] -> ['hi', "i'm", 'a', 'good', 'string']
    Some special cases are:
    [] -> []
    [''] -> ['']
    ['', ''] -> ['_']
    ['', '', '', ''] -> ['___']
    If reverse is true, empty strings are combined with closest non-empty string
    on the right or "" if it is the last string.
    """

    def _rev(l):
        return [s[::-1] for s in l][::-1]
    if reverse:
        return _rev(chomp_empty_strings(_rev(strings), c))
    if not len(strings):
        return strings
    if sum(map(len, strings)) == 0:
        return [c * (len(strings) - 1)]

    class _Chomper:

        def __init__(self, c):
            self.c = c

        def __call__(self, x, y):
            if len(y) == 0:
                return x[:-1] + [x[-1] + self.c]
            else:
                return x + [y]
    return list(filter(len, reduce(_Chomper(c), strings, [''])))