from __future__ import print_function
import re
import sys
import time
from pygments.filter import apply_filters, Filter
from pygments.filters import get_filter_by_name
from pygments.token import Error, Text, Other, _TokenType
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
from pygments.regexopt import regex_opt
def do_insertions(insertions, tokens):
    """
    Helper for lexers which must combine the results of several
    sublexers.

    ``insertions`` is a list of ``(index, itokens)`` pairs.
    Each ``itokens`` iterable should be inserted at position
    ``index`` into the token stream given by the ``tokens``
    argument.

    The result is a combined token stream.

    TODO: clean up the code here.
    """
    insertions = iter(insertions)
    try:
        index, itokens = next(insertions)
    except StopIteration:
        for item in tokens:
            yield item
        return
    realpos = None
    insleft = True
    for i, t, v in tokens:
        if realpos is None:
            realpos = i
        oldi = 0
        while insleft and i + len(v) >= index:
            tmpval = v[oldi:index - i]
            yield (realpos, t, tmpval)
            realpos += len(tmpval)
            for it_index, it_token, it_value in itokens:
                yield (realpos, it_token, it_value)
                realpos += len(it_value)
            oldi = index - i
            try:
                index, itokens = next(insertions)
            except StopIteration:
                insleft = False
                break
        yield (realpos, t, v[oldi:])
        realpos += len(v) - oldi
    while insleft:
        realpos = realpos or 0
        for p, t, v in itokens:
            yield (realpos, t, v)
            realpos += len(v)
        try:
            index, itokens = next(insertions)
        except StopIteration:
            insleft = False
            break