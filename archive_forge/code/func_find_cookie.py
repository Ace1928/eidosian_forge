from builtins import open as _builtin_open
from codecs import lookup, BOM_UTF8
import collections
import functools
from io import TextIOWrapper
import itertools as _itertools
import re
import sys
from token import *
from token import EXACT_TOKEN_TYPES
import token
def find_cookie(line):
    try:
        line_string = line.decode('utf-8')
    except UnicodeDecodeError:
        msg = 'invalid or missing encoding declaration'
        if filename is not None:
            msg = '{} for {!r}'.format(msg, filename)
        raise SyntaxError(msg)
    match = cookie_re.match(line_string)
    if not match:
        return None
    encoding = _get_normal_name(match.group(1))
    try:
        codec = lookup(encoding)
    except LookupError:
        if filename is None:
            msg = 'unknown encoding: ' + encoding
        else:
            msg = 'unknown encoding for {!r}: {}'.format(filename, encoding)
        raise SyntaxError(msg)
    if bom_found:
        if encoding != 'utf-8':
            if filename is None:
                msg = 'encoding problem: utf-8'
            else:
                msg = 'encoding problem for {!r}: utf-8'.format(filename)
            raise SyntaxError(msg)
        encoding += '-sig'
    return encoding