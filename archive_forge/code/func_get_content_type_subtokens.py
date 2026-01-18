import re
from pygments.lexer import RegexLexer, include
from pygments.lexers import get_lexer_for_mimetype
from pygments.token import Text, Name, String, Operator, Comment, Other
from pygments.util import get_int_opt, ClassNotFound
def get_content_type_subtokens(self, match):
    yield (match.start(1), Text, match.group(1))
    yield (match.start(2), Text.Whitespace, match.group(2))
    yield (match.start(3), Name.Attribute, match.group(3))
    yield (match.start(4), Operator, match.group(4))
    yield (match.start(5), String, match.group(5))
    if match.group(3).lower() == 'boundary':
        boundary = match.group(5).strip()
        if boundary[0] == '"' and boundary[-1] == '"':
            boundary = boundary[1:-1]
        self.boundary = boundary