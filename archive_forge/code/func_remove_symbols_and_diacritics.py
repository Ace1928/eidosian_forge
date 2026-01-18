import re
import unicodedata
from fractions import Fraction
from typing import Iterator, List, Match, Optional, Union
import regex
def remove_symbols_and_diacritics(s: str, keep=''):
    """
    Replace any other markers, symbols, and punctuations with a space, and drop any diacritics (category 'Mn' and some
    manual mappings)
    """

    def replace_character(char):
        if char in keep:
            return char
        elif char in ADDITIONAL_DIACRITICS:
            return ADDITIONAL_DIACRITICS[char]
        elif unicodedata.category(char) == 'Mn':
            return ''
        elif unicodedata.category(char)[0] in 'MSP':
            return ' '
        return char
    return ''.join((replace_character(c) for c in unicodedata.normalize('NFKD', s)))