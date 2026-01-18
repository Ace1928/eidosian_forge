from __future__ import annotations
import re
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, SupportsInt
import datetime
from collections.abc import Iterable
from babel import localtime
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
def format_weekday(self, char: str='E', num: int=4) -> str:
    """
        Return weekday from parsed datetime according to format pattern.

        >>> from datetime import date
        >>> format = DateTimeFormat(date(2016, 2, 28), Locale.parse('en_US'))
        >>> format.format_weekday()
        u'Sunday'

        'E': Day of week - Use one through three letters for the abbreviated day name, four for the full (wide) name,
             five for the narrow name, or six for the short name.
        >>> format.format_weekday('E',2)
        u'Sun'

        'e': Local day of week. Same as E except adds a numeric value that will depend on the local starting day of the
             week, using one or two letters. For this example, Monday is the first day of the week.
        >>> format.format_weekday('e',2)
        '01'

        'c': Stand-Alone local day of week - Use one letter for the local numeric value (same as 'e'), three for the
             abbreviated day name, four for the full (wide) name, five for the narrow name, or six for the short name.
        >>> format.format_weekday('c',1)
        '1'

        :param char: pattern format character ('e','E','c')
        :param num: count of format character

        """
    if num < 3:
        if char.islower():
            value = 7 - self.locale.first_week_day + self.value.weekday()
            return self.format(value % 7 + 1, num)
        num = 3
    weekday = self.value.weekday()
    width = {3: 'abbreviated', 4: 'wide', 5: 'narrow', 6: 'short'}[num]
    context = 'stand-alone' if char == 'c' else 'format'
    return get_day_names(width, context, self.locale)[weekday]