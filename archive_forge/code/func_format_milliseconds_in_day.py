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
def format_milliseconds_in_day(self, num):
    msecs = self.value.microsecond // 1000 + self.value.second * 1000 + self.value.minute * 60000 + self.value.hour * 3600000
    return self.format(msecs, num)