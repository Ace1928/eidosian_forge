import re
import html
from paste.util import PySourceColor
def str2html(src, strip=False, indent_subsequent=0, highlight_inner=False):
    """
    Convert a string to HTML.  Try to be really safe about it,
    returning a quoted version of the string if nothing else works.
    """
    try:
        return _str2html(src, strip=strip, indent_subsequent=indent_subsequent, highlight_inner=highlight_inner)
    except:
        return html_quote(src)