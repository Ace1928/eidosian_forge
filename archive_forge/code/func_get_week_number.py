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
def get_week_number(self, day_of_period: int, day_of_week: int | None=None) -> int:
    """Return the number of the week of a day within a period. This may be
        the week number in a year or the week number in a month.

        Usually this will return a value equal to or greater than 1, but if the
        first week of the period is so short that it actually counts as the last
        week of the previous period, this function will return 0.

        >>> date = datetime.date(2006, 1, 8)
        >>> DateTimeFormat(date, 'de_DE').get_week_number(6)
        1
        >>> DateTimeFormat(date, 'en_US').get_week_number(6)
        2

        :param day_of_period: the number of the day in the period (usually
                              either the day of month or the day of year)
        :param day_of_week: the week day; if omitted, the week day of the
                            current date is assumed
        """
    if day_of_week is None:
        day_of_week = self.value.weekday()
    first_day = (day_of_week - self.locale.first_week_day - day_of_period + 1) % 7
    if first_day < 0:
        first_day += 7
    week_number = (day_of_period + first_day - 1) // 7
    if 7 - first_day >= self.locale.min_week_days:
        week_number += 1
    if self.locale.first_week_day == 0:
        max_weeks = datetime.date(year=self.value.year, day=28, month=12).isocalendar()[1]
        if week_number > max_weeks:
            week_number -= max_weeks
    return week_number