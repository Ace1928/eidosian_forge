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
def format_quarter(self, char: str, num: int) -> str:
    quarter = (self.value.month - 1) // 3 + 1
    if num <= 2:
        return '%0*d' % (num, quarter)
    width = {3: 'abbreviated', 4: 'wide', 5: 'narrow'}[num]
    context = {'Q': 'format', 'q': 'stand-alone'}[char]
    return get_quarter_names(width, context, self.locale)[quarter]