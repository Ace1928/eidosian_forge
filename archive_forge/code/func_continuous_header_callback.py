import re
from pygments.lexer import RegexLexer, bygroups
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import ClassNotFound
def continuous_header_callback(self, match):
    yield (match.start(1), Text, match.group(1))
    yield (match.start(2), Literal, match.group(2))
    yield (match.start(3), Text, match.group(3))