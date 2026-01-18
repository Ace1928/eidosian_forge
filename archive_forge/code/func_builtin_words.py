import re
from pygments.lexer import bygroups, default, inherit, words
from pygments.lexers.lisp import SchemeLexer
from pygments.lexers._lilypond_builtins import (
from pygments.token import Token
def builtin_words(names, backslash, suffix=NAME_END_RE):
    prefix = '[\\-_^]?'
    if backslash == 'mandatory':
        prefix += '\\\\'
    elif backslash == 'optional':
        prefix += '\\\\?'
    else:
        assert backslash == 'disallowed'
    return words(names, prefix, suffix)