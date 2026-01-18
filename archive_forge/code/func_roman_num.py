import calendar
import datetime
import decimal
import re
from typing import Any, Iterator, List, Optional, Tuple, Union
from unicodedata import category
from ..exceptions import xpath_error
from ..regex import translate_pattern
from ._translation_maps import ALPHABET_CHARACTERS, OTHER_NUMBERS, ROMAN_NUMERALS_MAP, \
def roman_num(value: int) -> Iterator[str]:
    if not value:
        yield '0'
        return
    elif value < 0:
        yield '-'
        value = abs(value)
    for base, roman in ROMAN_NUMERALS_MAP.items():
        if value:
            yield (roman * (value // base))
            value %= base