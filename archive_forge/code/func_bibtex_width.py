from __future__ import absolute_import, unicode_literals
import re
import six
from pybtex.bibtex.exceptions import BibTeXError
from pybtex.py3compat import fix_unicode_literals_in_doctest
from pybtex.utils import pairwise
from pybtex import py3compat
def bibtex_width(string):
    """
    Determine the width of the given string, in relative units.

    >>> bibtex_width('')
    0
    >>> bibtex_width('abc')
    1500
    >>> bibtex_width('ab{c}')
    2500
    >>> bibtex_width(r"ab{\\'c}")
    1500
    >>> bibtex_width(r"ab{\\'c{}}")
    1500
    >>> bibtex_width(r"ab{\\'c{}")
    1500
    >>> bibtex_width(r"ab{\\'c{d}}")
    2056
    """
    from pybtex.charwidths import charwidths
    width = 0
    for token, brace_level in scan_bibtex_string(string):
        if brace_level == 1 and token.startswith('\\'):
            for char in token[2:]:
                if char not in '{}':
                    width += charwidths.get(char, 0)
            width -= 1000
        else:
            width += charwidths.get(token, 0)
    return width