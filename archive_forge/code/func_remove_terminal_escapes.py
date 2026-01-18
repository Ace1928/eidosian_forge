import codecs
import html
import re
import warnings
import ftfy
from ftfy.chardata import (
from ftfy.badness import is_bad
def remove_terminal_escapes(text):
    """
    Strip out "ANSI" terminal escape sequences, such as those that produce
    colored text on Unix.

        >>> print(remove_terminal_escapes(
        ...     "\\033[36;44mI'm blue, da ba dee da ba doo...\\033[0m"
        ... ))
        I'm blue, da ba dee da ba doo...
    """
    return ANSI_RE.sub('', text)