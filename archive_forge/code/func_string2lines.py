import sys
import re
import types
import unicodedata
from docutils import utils
from docutils.utils.error_reporting import ErrorOutput
def string2lines(astring, tab_width=8, convert_whitespace=False, whitespace=re.compile('[\x0b\x0c]')):
    """
    Return a list of one-line strings with tabs expanded, no newlines, and
    trailing whitespace stripped.

    Each tab is expanded with between 1 and `tab_width` spaces, so that the
    next character's index becomes a multiple of `tab_width` (8 by default).

    Parameters:

    - `astring`: a multi-line string.
    - `tab_width`: the number of columns between tab stops.
    - `convert_whitespace`: convert form feeds and vertical tabs to spaces?
    """
    if convert_whitespace:
        astring = whitespace.sub(' ', astring)
    return [s.expandtabs(tab_width).rstrip() for s in astring.splitlines()]