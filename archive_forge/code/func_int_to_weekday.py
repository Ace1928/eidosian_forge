import calendar
import datetime
import decimal
import re
from typing import Any, Iterator, List, Optional, Tuple, Union
from unicodedata import category
from ..exceptions import xpath_error
from ..regex import translate_pattern
from ._translation_maps import ALPHABET_CHARACTERS, OTHER_NUMBERS, ROMAN_NUMERALS_MAP, \
def int_to_weekday(num: int, lang: Optional[str]=None) -> str:
    if lang is None:
        lang = 'en'
    try:
        weekday_map = NUM_TO_WEEKDAY_MAPS[lang]
    except KeyError:
        weekday_map = NUM_TO_WEEKDAY_MAPS['en']
    return weekday_map[num]