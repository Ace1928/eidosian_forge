import re
import codecs
import sys
from docutils import nodes
from docutils.utils import split_escaped_whitespace, escape2null, unescape
from docutils.parsers.rst.languages import en as _fallback_language_module
def positive_int(argument):
    """
    Converts the argument into an integer.  Raises ValueError for negative,
    zero, or non-integer values.  (Directive option conversion function.)
    """
    value = int(argument)
    if value < 1:
        raise ValueError('negative or zero value; must be positive')
    return value