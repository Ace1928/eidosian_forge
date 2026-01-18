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
def get_datetime_format(format: _PredefinedTimeFormat='medium', locale: Locale | str | None=LC_TIME) -> DateTimePattern:
    """Return the datetime formatting patterns used by the locale for the
    specified format.

    >>> get_datetime_format(locale='en_US')
    u'{1}, {0}'

    :param format: the format to use, one of "full", "long", "medium", or
                   "short"
    :param locale: the `Locale` object, or a locale string
    """
    patterns = Locale.parse(locale).datetime_formats
    if format not in patterns:
        format = None
    return patterns[format]