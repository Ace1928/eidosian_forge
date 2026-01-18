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
def get_era_names(width: Literal['abbreviated', 'narrow', 'wide']='wide', locale: Locale | str | None=LC_TIME) -> LocaleDataDict:
    """Return the era names used by the locale for the specified format.

    >>> get_era_names('wide', locale='en_US')[1]
    u'Anno Domini'
    >>> get_era_names('abbreviated', locale='de_DE')[1]
    u'n. Chr.'

    :param width: the width to use, either "wide", "abbreviated", or "narrow"
    :param locale: the `Locale` object, or a locale string
    """
    return Locale.parse(locale).eras[width]