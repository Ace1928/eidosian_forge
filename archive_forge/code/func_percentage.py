import re
import codecs
import sys
from docutils import nodes
from docutils.utils import split_escaped_whitespace, escape2null, unescape
from docutils.parsers.rst.languages import en as _fallback_language_module
def percentage(argument):
    """
    Check for an integer percentage value with optional percent sign.
    """
    try:
        argument = argument.rstrip(' %')
    except AttributeError:
        pass
    return nonnegative_int(argument)