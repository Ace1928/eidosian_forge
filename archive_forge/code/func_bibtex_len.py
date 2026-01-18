from __future__ import absolute_import, unicode_literals
import re
import six
from pybtex.bibtex.exceptions import BibTeXError
from pybtex.py3compat import fix_unicode_literals_in_doctest
from pybtex.utils import pairwise
from pybtex import py3compat
def bibtex_len(string):
    """Return the number of characters in the string.

    Braces are ignored. "Special characters" are ignored. A "special character"
    is a substring at brace level 1, if the first character after the opening
    brace is a backslash, like in "de la Vall{\\'e}e Poussin".

    >>> print(bibtex_len(r"de la Vall{\\'e}e Poussin"))
    20
    >>> print(bibtex_len(r"de la Vall{e}e Poussin"))
    20
    >>> print(bibtex_len(r"de la Vallee Poussin"))
    20
    >>> print(bibtex_len(r'\\ABC 123'))
    8
    >>> print(bibtex_len(r'{\\abc}'))
    1
    >>> print(bibtex_len(r'{\\abc'))
    1
    >>> print(bibtex_len(r'}\\abc'))
    4
    >>> print(bibtex_len(r'\\abc}'))
    4
    >>> print(bibtex_len(r'\\abc{'))
    4
    >>> print(bibtex_len(r'level 0 {1 {2}}'))
    11
    >>> print(bibtex_len(r'level 0 {\\1 {2}}'))
    9
    >>> print(bibtex_len(r'level 0 {1 {\\2}}'))
    12
    """
    length = 0
    for char, brace_level in scan_bibtex_string(string):
        if char not in '{}':
            length += 1
    return length