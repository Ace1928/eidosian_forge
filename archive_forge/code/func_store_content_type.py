import re
from pygments.lexer import RegexLexer, include
from pygments.lexers import get_lexer_for_mimetype
from pygments.token import Text, Name, String, Operator, Comment, Other
from pygments.util import get_int_opt, ClassNotFound
def store_content_type(self, match):
    self.content_type = match.group(1)
    prefix_len = match.start(1) - match.start(0)
    yield (match.start(0), Text.Whitespace, match.group(0)[:prefix_len])
    yield (match.start(1), Name.Label, match.group(2))
    yield (match.end(2), String.Delimiter, '/')
    yield (match.start(3), Name.Label, match.group(3))