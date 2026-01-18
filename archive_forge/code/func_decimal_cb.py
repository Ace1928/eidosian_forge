import re
from pygments.lexer import RegexLexer, include, bygroups, words, default
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.python import PythonLexer
from pygments.lexers._scheme_builtins import scheme_keywords, scheme_builtins
def decimal_cb(self, match):
    if '.' in match.group():
        token_type = Number.Float
    else:
        token_type = Number.Integer
    yield (match.start(), token_type, match.group())