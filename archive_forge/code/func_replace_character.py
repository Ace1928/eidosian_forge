import re
import unicodedata
from fractions import Fraction
from typing import Iterator, List, Match, Optional, Union
import regex
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