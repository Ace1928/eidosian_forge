import abc
import functools
import re
import tokenize
from tokenize import untokenize, TokenError
from io import StringIO
from IPython.core.splitinput import LineInfo
from IPython.utils import tokenutil
def ends_in_comment_or_string(src):
    """Indicates whether or not an input line ends in a comment or within
    a multiline string.

    Parameters
    ----------
    src : string
        A single line input string.

    Returns
    -------
    comment : bool
        True if source ends in a comment or multiline string.
    """
    toktypes = _line_tokens(src)
    return tokenize.COMMENT in toktypes or _MULTILINE_STRING in toktypes