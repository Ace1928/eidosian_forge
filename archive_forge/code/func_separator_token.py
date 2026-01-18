import operator
from debian._deb822_repro._util import BufferingIterator
from debian._deb822_repro.tokens import Deb822Token
@classmethod
def separator_token(cls, text):
    if text == ' ':
        return SPACE_SEPARATOR_FT
    if text == ',':
        return COMMA_SEPARATOR_FT
    return cls(text, _CONTENT_TYPE_SEPARATOR)