import re
import codecs
import sys
from docutils import nodes
from docutils.utils import split_escaped_whitespace, escape2null, unescape
from docutils.parsers.rst.languages import en as _fallback_language_module
def single_char_or_whitespace_or_unicode(argument):
    """
    As with `single_char_or_unicode`, but "tab" and "space" are also supported.
    (Directive option conversion function.)
    """
    if argument == 'tab':
        char = '\t'
    elif argument == 'space':
        char = ' '
    else:
        char = single_char_or_unicode(argument)
    return char