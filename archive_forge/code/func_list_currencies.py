from __future__ import annotations
import datetime
import decimal
import re
import warnings
from typing import TYPE_CHECKING, Any, cast, overload
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
def list_currencies(locale: Locale | str | None=None) -> set[str]:
    """ Return a `set` of normalized currency codes.

    .. versionadded:: 2.5.0

    :param locale: filters returned currency codes by the provided locale.
                   Expected to be a locale instance or code. If no locale is
                   provided, returns the list of all currencies from all
                   locales.
    """
    if locale:
        return set(Locale.parse(locale).currencies)
    return set(get_global('all_currencies'))