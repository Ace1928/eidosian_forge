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
def get_day_of_year(self, date: datetime.date | None=None) -> int:
    if date is None:
        date = self.value
    return (date - date.replace(month=1, day=1)).days + 1