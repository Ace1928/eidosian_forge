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
def get_day_names(width: Literal['abbreviated', 'narrow', 'short', 'wide']='wide', context: _Context='format', locale: Locale | str | None=LC_TIME) -> LocaleDataDict:
    """Return the day names used by the locale for the specified format.

    >>> get_day_names('wide', locale='en_US')[1]
    u'Tuesday'
    >>> get_day_names('short', locale='en_US')[1]
    u'Tu'
    >>> get_day_names('abbreviated', locale='es')[1]
    u'mar'
    >>> get_day_names('narrow', context='stand-alone', locale='de_DE')[1]
    u'D'

    :param width: the width to use, one of "wide", "abbreviated", "short" or "narrow"
    :param context: the context, either "format" or "stand-alone"
    :param locale: the `Locale` object, or a locale string
    """
    return Locale.parse(locale).days[context][width]