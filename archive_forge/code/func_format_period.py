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
def format_period(self, char: str, num: int) -> str:
    """
        Return period from parsed datetime according to format pattern.

        >>> from datetime import datetime, time
        >>> format = DateTimeFormat(time(13, 42), 'fi_FI')
        >>> format.format_period('a', 1)
        u'ip.'
        >>> format.format_period('b', 1)
        u'iltap.'
        >>> format.format_period('b', 4)
        u'iltapäivä'
        >>> format.format_period('B', 4)
        u'iltapäivällä'
        >>> format.format_period('B', 5)
        u'ip.'

        >>> format = DateTimeFormat(datetime(2022, 4, 28, 6, 27), 'zh_Hant')
        >>> format.format_period('a', 1)
        u'上午'
        >>> format.format_period('b', 1)
        u'清晨'
        >>> format.format_period('B', 1)
        u'清晨'

        :param char: pattern format character ('a', 'b', 'B')
        :param num: count of format character

        """
    widths = [{3: 'abbreviated', 4: 'wide', 5: 'narrow'}[max(3, num)], 'wide', 'narrow', 'abbreviated']
    if char == 'a':
        period = 'pm' if self.value.hour >= 12 else 'am'
        context = 'format'
    else:
        period = get_period_id(self.value, locale=self.locale)
        context = 'format' if char == 'B' else 'stand-alone'
    for width in widths:
        period_names = get_period_names(context=context, width=width, locale=self.locale)
        if period in period_names:
            return period_names[period]
    raise ValueError(f'Could not format period {period} in {self.locale}')