import re
import html
from paste.util import PySourceColor
def pretty_string_repr(self, s):
    """
        Formats the string as a triple-quoted string when it contains
        newlines.
        """
    if '\n' in s:
        s = repr(s)
        s = s[0] * 3 + s[1:-1] + s[-1] * 3
        s = s.replace('\\n', '\n')
        return s
    else:
        return repr(s)