from __future__ import annotations
import datetime
import decimal
import re
import warnings
from typing import TYPE_CHECKING, Any, cast, overload
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
def get_currency_precision(currency: str) -> int:
    """Return currency's precision.

    Precision is the number of decimals found after the decimal point in the
    currency's format pattern.

    .. versionadded:: 2.5.0

    :param currency: the currency code.
    """
    precisions = get_global('currency_fractions')
    return precisions.get(currency, precisions['DEFAULT'])[0]