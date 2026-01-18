import re
from pygments.lexer import RegexLexer, include, bygroups, default, words
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
def long_id_callback(self, match):
    if match.group(1) in self.alphanumid_reserved:
        token = Error
    else:
        token = Name.Namespace
    yield (match.start(1), token, match.group(1))
    yield (match.start(2), Punctuation, match.group(2))