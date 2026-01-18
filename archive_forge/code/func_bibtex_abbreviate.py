from __future__ import absolute_import, unicode_literals
import re
import six
from pybtex.bibtex.exceptions import BibTeXError
from pybtex.py3compat import fix_unicode_literals_in_doctest
from pybtex.utils import pairwise
from pybtex import py3compat
def bibtex_abbreviate(string, delimiter=None, separator='-'):
    """
    Abbreviate string.

    >>> print(bibtex_abbreviate('Andrew Blake'))
    A
    >>> print(bibtex_abbreviate('Jean-Pierre'))
    J.-P
    >>> print(bibtex_abbreviate('Jean--Pierre'))
    J.-P
    
    """

    def _bibtex_abbreviate():
        for token in split_tex_string(string, sep=separator):
            letter = bibtex_first_letter(token)
            if letter:
                yield letter
    if delimiter is None:
        delimiter = '.-'
    return delimiter.join(_bibtex_abbreviate())