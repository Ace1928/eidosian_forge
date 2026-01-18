from __future__ import annotations
import datetime
import decimal
import re
import warnings
from typing import TYPE_CHECKING, Any, cast, overload
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
def get_currency_symbol(currency: str, locale: Locale | str | None=LC_NUMERIC) -> str:
    """Return the symbol used by the locale for the specified currency.

    >>> get_currency_symbol('USD', locale='en_US')
    u'$'

    :param currency: the currency code.
    :param locale: the `Locale` object or locale identifier.
    """
    return Locale.parse(locale).currency_symbols.get(currency, currency)