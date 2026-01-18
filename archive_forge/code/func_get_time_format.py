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
def get_time_format(format: _PredefinedTimeFormat='medium', locale: Locale | str | None=LC_TIME) -> DateTimePattern:
    """Return the time formatting patterns used by the locale for the specified
    format.

    >>> get_time_format(locale='en_US')
    <DateTimePattern u'h:mm:ss\u202fa'>
    >>> get_time_format('full', locale='de_DE')
    <DateTimePattern u'HH:mm:ss zzzz'>

    :param format: the format to use, one of "full", "long", "medium", or
                   "short"
    :param locale: the `Locale` object, or a locale string
    """
    return Locale.parse(locale).time_formats[format]