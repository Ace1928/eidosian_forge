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
def get_quarter_names(width: Literal['abbreviated', 'narrow', 'wide']='wide', context: _Context='format', locale: Locale | str | None=LC_TIME) -> LocaleDataDict:
    """Return the quarter names used by the locale for the specified format.

    >>> get_quarter_names('wide', locale='en_US')[1]
    u'1st quarter'
    >>> get_quarter_names('abbreviated', locale='de_DE')[1]
    u'Q1'
    >>> get_quarter_names('narrow', locale='de_DE')[1]
    u'1'

    :param width: the width to use, one of "wide", "abbreviated", or "narrow"
    :param context: the context, either "format" or "stand-alone"
    :param locale: the `Locale` object, or a locale string
    """
    return Locale.parse(locale).quarters[context][width]