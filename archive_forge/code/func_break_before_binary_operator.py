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
def break_before_binary_operator(logical_line, tokens):
    """
    Avoid breaks before binary operators.

    The preferred place to break around a binary operator is after the
    operator, not before it.

    W503: (width == 0\\n + height == 0)
    W503: (width == 0\\n and height == 0)
    W503: var = (1\\n       & ~2)
    W503: var = (1\\n       / -2)
    W503: var = (1\\n       + -1\\n       + -2)

    Okay: foo(\\n    -x)
    Okay: foo(x\\n    [])
    Okay: x = '''\\n''' + ''
    Okay: foo(x,\\n    -y)
    Okay: foo(x,  # comment\\n    -y)
    """
    for context in _break_around_binary_operators(tokens):
        token_type, text, previous_token_type, previous_text, line_break, unary_context, start = context
        if _is_binary_operator(token_type, text) and line_break and (not unary_context) and (not _is_binary_operator(previous_token_type, previous_text)):
            yield (start, 'W503 line break before binary operator')