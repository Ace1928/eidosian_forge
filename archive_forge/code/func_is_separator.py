import operator
from debian._deb822_repro._util import BufferingIterator
from debian._deb822_repro.tokens import Deb822Token
@property
def is_separator(self):
    """True if this formatter token represents a separator token

        The formatter is not required to preserve the provided separators but it
        is required to properly separate values.  In fact, often is a lot easier
        to discard existing separator tokens.  As an example, in whitespace
        separated list of values space, tab and newline all counts as separator.
        However, formatting-wise, there is a world of difference between the
        a space, tab and a newline. In particularly, newlines must be followed
        by an additional space or tab (to act as a value continuation line) if
        there is a value following it (otherwise, the generated output is
        invalid).
        """
    return self._content_type is _CONTENT_TYPE_SEPARATOR