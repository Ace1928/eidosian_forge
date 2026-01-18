import bisect
import configparser
import inspect
import io
import keyword
import os
import re
import sys
import time
import tokenize
import warnings
from fnmatch import fnmatch
from functools import lru_cache
from optparse import OptionParser
@register_check
def break_after_binary_operator(logical_line, tokens):
    """
    Avoid breaks after binary operators.

    The preferred place to break around a binary operator is before the
    operator, not after it.

    W504: (width == 0 +\\n height == 0)
    W504: (width == 0 and\\n height == 0)
    W504: var = (1 &\\n       ~2)

    Okay: foo(\\n    -x)
    Okay: foo(x\\n    [])
    Okay: x = '''\\n''' + ''
    Okay: x = '' + '''\\n'''
    Okay: foo(x,\\n    -y)
    Okay: foo(x,  # comment\\n    -y)

    The following should be W504 but unary_context is tricky with these
    Okay: var = (1 /\\n       -2)
    Okay: var = (1 +\\n       -1 +\\n       -2)
    """
    prev_start = None
    for context in _break_around_binary_operators(tokens):
        token_type, text, previous_token_type, previous_text, line_break, unary_context, start = context
        if _is_binary_operator(previous_token_type, previous_text) and line_break and (not unary_context) and (not _is_binary_operator(token_type, text)):
            yield (prev_start, 'W504 line break after binary operator')
        prev_start = start