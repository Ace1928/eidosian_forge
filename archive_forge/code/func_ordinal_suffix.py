import calendar
import datetime
import decimal
import re
from typing import Any, Iterator, List, Optional, Tuple, Union
from unicodedata import category
from ..exceptions import xpath_error
from ..regex import translate_pattern
from ._translation_maps import ALPHABET_CHARACTERS, OTHER_NUMBERS, ROMAN_NUMERALS_MAP, \
def ordinal_suffix(value: int) -> str:
    value = abs(value) % 100
    if 3 < value < 20:
        return 'th'
    value %= 10
    if value == 1:
        return 'st'
    elif value == 2:
        return 'nd'
    elif value == 3:
        return 'rd'
    else:
        return 'th'