from __future__ import annotations
import decimal
import gettext
import locale
import os
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Callable, Iterable
from babel.core import Locale
from babel.dates import format_date, format_datetime, format_time, format_timedelta
from babel.numbers import (
def compact_currency(self, number: float | decimal.Decimal | str, currency: str, format_type: Literal['short']='short', fraction_digits: int=0) -> str:
    """Return a number in the given currency formatted for the locale
        using the compact number format.

        >>> Format('en_US').compact_currency(1234567, "USD", format_type='short', fraction_digits=2)
        '$1.23M'
        """
    return format_compact_currency(number, currency, format_type=format_type, fraction_digits=fraction_digits, locale=self.locale, numbering_system=self.numbering_system)