import re
from pygments.lexer import RegexLexer, include
from pygments.lexers import get_lexer_for_mimetype
from pygments.token import Text, Name, String, Operator, Comment, Other
from pygments.util import get_int_opt, ClassNotFound
def get_header_tokens(self, match):
    field = match.group(1)
    if field.lower() in self.attention_headers:
        yield (match.start(1), Name.Tag, field + ':')
        yield (match.start(2), Text.Whitespace, match.group(2))
        pos = match.end(2)
        body = match.group(3)
        for i, t, v in self.get_tokens_unprocessed(body, ('root', field.lower())):
            yield (pos + i, t, v)
    else:
        yield (match.start(), Comment, match.group())