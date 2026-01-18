import re
from pygments.lexer import RegexLexer, include
from pygments.lexers import get_lexer_for_mimetype
from pygments.token import Text, Name, String, Operator, Comment, Other
from pygments.util import get_int_opt, ClassNotFound
def store_content_transfer_encoding(self, match):
    self.content_transfer_encoding = match.group(0).lower()
    yield (match.start(0), Name.Constant, match.group(0))