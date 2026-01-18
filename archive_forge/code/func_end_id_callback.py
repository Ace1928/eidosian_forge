import re
from pygments.lexer import RegexLexer, include, bygroups, default, words
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
def end_id_callback(self, match):
    if match.group(1) in self.alphanumid_reserved:
        token = Error
    elif match.group(1) in self.symbolicid_reserved:
        token = Error
    else:
        token = Name
    yield (match.start(1), token, match.group(1))