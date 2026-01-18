from __future__ import annotations
import datetime
import decimal
import re
import warnings
from typing import TYPE_CHECKING, Any, cast, overload
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
def get_decimal_quantum(precision: int | decimal.Decimal) -> decimal.Decimal:
    """Return minimal quantum of a number, as defined by precision."""
    assert isinstance(precision, (int, decimal.Decimal))
    return decimal.Decimal(10) ** (-precision)