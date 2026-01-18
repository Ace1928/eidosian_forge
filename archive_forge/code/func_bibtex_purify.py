from __future__ import absolute_import, unicode_literals
import re
import six
from pybtex.bibtex.exceptions import BibTeXError
from pybtex.py3compat import fix_unicode_literals_in_doctest
from pybtex.utils import pairwise
from pybtex import py3compat
def bibtex_purify(string):
    """Strip special characters from the string.

    >>> print(bibtex_purify('Abc 1234'))
    Abc 1234
    >>> print(bibtex_purify('Abc  1234'))
    Abc  1234
    >>> print(bibtex_purify('Abc-Def'))
    Abc Def
    >>> print(bibtex_purify('Abc-~-Def'))
    Abc   Def
    >>> print(bibtex_purify('{XXX YYY}'))
    XXX YYY
    >>> print(bibtex_purify('{XXX {YYY}}'))
    XXX YYY
    >>> print(bibtex_purify(r'XXX {\\YYY} XXX'))
    XXX  XXX
    >>> print(bibtex_purify(r'{XXX {\\YYY} XXX}'))
    XXX YYY XXX
    >>> print(bibtex_purify(r'\\\\abc def'))
    abc def
    >>> print(bibtex_purify('a@#$@#$b@#$@#$c'))
    abc
    >>> print(bibtex_purify(r'{\\noopsort{1973b}}1973'))
    1973b1973
    >>> print(bibtex_purify(r'{sort{1973b}}1973'))
    sort1973b1973
    >>> print(bibtex_purify(r'{sort{\\abc1973b}}1973'))
    sortabc1973b1973
    >>> print(bibtex_purify(r'{\\noopsort{1973a}}{\\switchargs{--90}{1968}}'))
    1973a901968
    """

    def purify_iter(string):
        for token, brace_level in scan_bibtex_string(string):
            if brace_level == 1 and token.startswith('\\'):
                for char in purify_special_char_re.sub('', token):
                    if char.isalnum():
                        yield char
            elif token.isalnum():
                yield token
            elif token.isspace() or token in '-~':
                yield ' '
    return ''.join(purify_iter(string))