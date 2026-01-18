import codecs
import html
import re
import warnings
import ftfy
from ftfy.chardata import (
from ftfy.badness import is_bad
def fix_line_breaks(text):
    """
    Convert all line breaks to Unix style.

    This will convert the following sequences into the standard \\\\n
    line break:

    - CRLF (\\\\r\\\\n), used on Windows and in some communication protocols
    - CR (\\\\r), once used on Mac OS Classic, and now kept alive by misguided
      software such as Microsoft Office for Mac
    - LINE SEPARATOR (\\\\u2028) and PARAGRAPH SEPARATOR (\\\\u2029), defined by
      Unicode and used to sow confusion and discord
    - NEXT LINE (\\\\x85), a C1 control character that is certainly not what you
      meant

    The NEXT LINE character is a bit of an odd case, because it
    usually won't show up if `fix_encoding` is also being run.
    \\\\x85 is very common mojibake for \\\\u2026, HORIZONTAL ELLIPSIS.

        >>> print(fix_line_breaks(
        ...     "This string is made of two things:\\u2029"
        ...     "1. Unicode\\u2028"
        ...     "2. Spite"
        ... ))
        This string is made of two things:
        1. Unicode
        2. Spite

    For further testing and examples, let's define a function to make sure
    we can see the control characters in their escaped form:

        >>> def eprint(text):
        ...     print(text.encode('unicode-escape').decode('ascii'))

        >>> eprint(fix_line_breaks("Content-type: text/plain\\r\\n\\r\\nHi."))
        Content-type: text/plain\\n\\nHi.

        >>> eprint(fix_line_breaks("This is how Microsoft \\r trolls Mac users"))
        This is how Microsoft \\n trolls Mac users

        >>> eprint(fix_line_breaks("What is this \\x85 I don't even"))
        What is this \\n I don't even
    """
    return text.replace('\r\n', '\n').replace('\r', '\n').replace('\u2028', '\n').replace('\u2029', '\n').replace('\x85', '\n')