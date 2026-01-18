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
def compact_decimal(self, number: float | decimal.Decimal | str, format_type: Literal['short', 'long']='short', fraction_digits: int=0) -> str:
    """Return a number formatted in compact form for the locale.

        >>> fmt = Format('en_US')
        >>> fmt.compact_decimal(123456789)
        u'123M'
        >>> fmt.compact_decimal(1234567, format_type='long', fraction_digits=2)
        '1.23 million'
        """
    return format_compact_decimal(number, format_type=format_type, fraction_digits=fraction_digits, locale=self.locale, numbering_system=self.numbering_system)