import operator
from debian._deb822_repro._util import BufferingIterator
from debian._deb822_repro.tokens import Deb822Token
@classmethod
def comment_token(cls, text):
    """Generates a single comment token with the provided text

        Mostly useful for creating test cases
        """
    return cls(text, _CONTENT_TYPE_COMMENT)