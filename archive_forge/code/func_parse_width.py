import calendar
import datetime
import decimal
import re
from typing import Any, Iterator, List, Optional, Tuple, Union
from unicodedata import category
from ..exceptions import xpath_error
from ..regex import translate_pattern
from ._translation_maps import ALPHABET_CHARACTERS, OTHER_NUMBERS, ROMAN_NUMERALS_MAP, \
def parse_width(width: str) -> Tuple[int, Optional[int]]:
    min_width: Union[str, int]
    max_width: Union[str, int, None]
    if WIDTH_PATTERN.match(width) is None:
        raise xpath_error('FOFD1340', f'Invalid width modifier {width!r}')
    elif '-' not in width:
        if width == '*':
            return (0, None)
        min_width = int(width)
        if not min_width:
            raise xpath_error('FOFD1340', f'Invalid width modifier {width!r}')
        return (min_width, None)
    elif '*' not in width:
        min_width, max_width = map(int, width.split('-'))
        if not min_width or max_width < min_width:
            raise xpath_error('FOFD1340', f'Invalid width modifier {width!r}')
        return (min_width, max_width)
    else:
        min_width, max_width = width.split('-')
        if min_width == '*':
            min_width = 0
        else:
            min_width = int(min_width)
            if not min_width:
                raise xpath_error('FOFD1340', f'Invalid width modifier {width!r}')
        if max_width == '*':
            return (min_width, None)
        else:
            max_width = int(max_width)
            if not max_width:
                raise xpath_error('FOFD1340', f'Invalid width modifier {width!r}')
            return (min_width, max_width)