import re
import codecs
import sys
from docutils import nodes
from docutils.utils import split_escaped_whitespace, escape2null, unescape
from docutils.parsers.rst.languages import en as _fallback_language_module
def single_char_or_unicode(argument):
    """
    A single character is returned as-is.  Unicode characters codes are
    converted as in `unicode_code`.  (Directive option conversion function.)
    """
    char = unicode_code(argument)
    if len(char) > 1:
        raise ValueError('%r invalid; must be a single character or a Unicode code' % char)
    return char