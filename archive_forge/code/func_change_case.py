from __future__ import absolute_import, unicode_literals
import re
import six
from pybtex.bibtex.exceptions import BibTeXError
from pybtex.py3compat import fix_unicode_literals_in_doctest
from pybtex.utils import pairwise
from pybtex import py3compat
def change_case(string, mode):
    """
    >>> print(change_case('aBcD', 'l'))
    abcd
    >>> print(change_case('aBcD', 'u'))
    ABCD
    >>> print(change_case('ABcD', 't'))
    Abcd
    >>> print(change_case(r'The {\\TeX book \\noop}', 'u'))
    THE {\\TeX BOOK \\noop}
    >>> print(change_case(r'And Now: BOOO!!!', 't'))
    And now: Booo!!!
    >>> print(change_case(r'And {Now: BOOO!!!}', 't'))
    And {Now: BOOO!!!}
    >>> print(change_case(r'And {Now: {BOOO}!!!}', 'l'))
    and {Now: {BOOO}!!!}
    >>> print(change_case(r'And {\\Now: BOOO!!!}', 't'))
    And {\\Now: booo!!!}
    >>> print(change_case(r'And {\\Now: {BOOO}!!!}', 'l'))
    and {\\Now: {booo}!!!}
    >>> print(change_case(r'{\\TeX\\ and databases\\Dash\\TeX DBI}', 't'))
    {\\TeX\\ and databases\\Dash\\TeX DBI}
    """

    def title(char, state):
        if state == 'start':
            return char
        else:
            return char.lower()
    lower = lambda char, state: char.lower()
    upper = lambda char, state: char.upper()
    convert = {'l': lower, 'u': upper, 't': title}[mode]

    def convert_special_char(special_char, state):

        def convert_words(words):
            for word in words:
                if word.startswith('\\'):
                    yield word
                else:
                    yield convert(word, state)
        return ' '.join(convert_words(special_char.split(' ')))

    def change_case_iter(string, mode):
        state = 'start'
        for char, brace_level in scan_bibtex_string(string):
            if brace_level == 0:
                yield convert(char, state)
                if char == ':':
                    state = 'after colon'
                elif char.isspace() and state == 'after colon':
                    state = 'start'
                else:
                    state = 'normal'
            elif brace_level == 1 and char.startswith('\\'):
                yield convert_special_char(char, state)
            else:
                yield char
    return ''.join(change_case_iter(string, mode))