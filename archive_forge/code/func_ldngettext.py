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
def ldngettext(self, domain: str, singular: str, plural: str, num: int) -> str:
    """Like ``lngettext()``, but look the message up in the specified
        domain.
        """
    import warnings
    warnings.warn('ldngettext() is deprecated, use dngettext() instead', DeprecationWarning, stacklevel=2)
    return self._domains.get(domain, self).lngettext(singular, plural, num)