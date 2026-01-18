import operator
from debian._deb822_repro._util import BufferingIterator
from debian._deb822_repro.tokens import Deb822Token
@property
def is_value(self):
    """True if this formatter token represents a semantic value

        The formatter *MUST* preserve values as-in in its output.  It may
        "unpack" it from the token (as in, return it as a part of a plain
        str) but the value content must not be changed nor re-ordered relative
        to other value tokens (as that could change the meaning of the field).
        """
    return self._content_type is _CONTENT_TYPE_VALUE