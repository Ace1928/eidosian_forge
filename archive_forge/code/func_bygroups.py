from __future__ import print_function
import re
import sys
import time
from pygments.filter import apply_filters, Filter
from pygments.filters import get_filter_by_name
from pygments.token import Error, Text, Other, _TokenType
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
from pygments.regexopt import regex_opt
def bygroups(*args):
    """
    Callback that yields multiple actions for each group in the match.
    """

    def callback(lexer, match, ctx=None):
        for i, action in enumerate(args):
            if action is None:
                continue
            elif type(action) is _TokenType:
                data = match.group(i + 1)
                if data:
                    yield (match.start(i + 1), action, data)
            else:
                data = match.group(i + 1)
                if data is not None:
                    if ctx:
                        ctx.pos = match.start(i + 1)
                    for item in action(lexer, _PseudoMatch(match.start(i + 1), data), ctx):
                        if item:
                            yield item
        if ctx:
            ctx.pos = match.end()
    return callback