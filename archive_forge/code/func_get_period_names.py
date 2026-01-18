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
def get_period_names(width: Literal['abbreviated', 'narrow', 'wide']='wide', context: _Context='stand-alone', locale: Locale | str | None=LC_TIME) -> LocaleDataDict:
    """Return the names for day periods (AM/PM) used by the locale.

    >>> get_period_names(locale='en_US')['am']
    u'AM'

    :param width: the width to use, one of "abbreviated", "narrow", or "wide"
    :param context: the context, either "format" or "stand-alone"
    :param locale: the `Locale` object, or a locale string
    """
    return Locale.parse(locale).day_periods[context][width]