import operator
from debian._deb822_repro._util import BufferingIterator
from debian._deb822_repro.tokens import Deb822Token
@classmethod
def from_token_or_element(cls, token_or_element):
    if isinstance(token_or_element, Deb822Token):
        if token_or_element.is_comment:
            return cls.comment_token(token_or_element.text)
        if token_or_element.is_whitespace:
            raise ValueError('FormatterContentType cannot be whitespace')
        return cls.value_token(token_or_element.text)
    return cls.value_token(token_or_element.convert_to_text())