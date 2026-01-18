import re
from pygments.lexer import RegexLexer, bygroups
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import ClassNotFound
def header_callback(self, match):
    if match.group(1).lower() == 'content-type':
        content_type = match.group(5).strip()
        if ';' in content_type:
            content_type = content_type[:content_type.find(';')].strip()
        self.content_type = content_type
    yield (match.start(1), Name.Attribute, match.group(1))
    yield (match.start(2), Text, match.group(2))
    yield (match.start(3), Operator, match.group(3))
    yield (match.start(4), Text, match.group(4))
    yield (match.start(5), Literal, match.group(5))
    yield (match.start(6), Text, match.group(6))