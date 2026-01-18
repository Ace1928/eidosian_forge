from __future__ import absolute_import, unicode_literals
import re
import six
from pybtex.bibtex.exceptions import BibTeXError
from pybtex.py3compat import fix_unicode_literals_in_doctest
from pybtex.utils import pairwise
from pybtex import py3compat
def bibtex_prefix(string, num_chars):
    """Return the firxt num_char characters of the string.

    Braces and "special characters" are ignored, as in bibtex_len.  If the
    resulting prefix ends at brace level > 0, missing closing braces are
    appended.

    >>> print(bibtex_prefix('abc', 1))
    a
    >>> print(bibtex_prefix('abc', 5))
    abc
    >>> print(bibtex_prefix('ab{c}d', 3))
    ab{c}
    >>> print(bibtex_prefix('ab{cd}', 3))
    ab{c}
    >>> print(bibtex_prefix('ab{cd', 3))
    ab{c}
    >>> print(bibtex_prefix(r'ab{\\cd}', 3))
    ab{\\cd}
    >>> print(bibtex_prefix(r'ab{\\cd', 3))
    ab{\\cd}

    """

    def prefix():
        length = 0
        for char, brace_level in scan_bibtex_string(string):
            yield char
            if char not in '{}':
                length += 1
            if length >= num_chars:
                break
        for i in range(brace_level):
            yield '}'
    return ''.join(prefix())