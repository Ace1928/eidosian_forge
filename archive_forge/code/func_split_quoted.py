import importlib.util
import os
import re
import string
import subprocess
import sys
import sysconfig
import functools
from .errors import DistutilsPlatformError, DistutilsByteCompileError
from ._modified import newer
from .spawn import spawn
from ._log import log
from distutils.util import byte_compile
def split_quoted(s):
    """Split a string up according to Unix shell-like rules for quotes and
    backslashes.  In short: words are delimited by spaces, as long as those
    spaces are not escaped by a backslash, or inside a quoted string.
    Single and double quotes are equivalent, and the quote characters can
    be backslash-escaped.  The backslash is stripped from any two-character
    escape sequence, leaving only the escaped character.  The quote
    characters are stripped from any quoted string.  Returns a list of
    words.
    """
    if _wordchars_re is None:
        _init_regex()
    s = s.strip()
    words = []
    pos = 0
    while s:
        m = _wordchars_re.match(s, pos)
        end = m.end()
        if end == len(s):
            words.append(s[:end])
            break
        if s[end] in string.whitespace:
            words.append(s[:end])
            s = s[end:].lstrip()
            pos = 0
        elif s[end] == '\\':
            s = s[:end] + s[end + 1:]
            pos = end + 1
        else:
            if s[end] == "'":
                m = _squote_re.match(s, end)
            elif s[end] == '"':
                m = _dquote_re.match(s, end)
            else:
                raise RuntimeError("this can't happen (bad char '%c')" % s[end])
            if m is None:
                raise ValueError('bad string (mismatched %s quotes?)' % s[end])
            beg, end = m.span()
            s = s[:beg] + s[beg + 1:end - 1] + s[end:]
            pos = m.end() - 2
        if pos >= len(s):
            words.append(s)
            break
    return words