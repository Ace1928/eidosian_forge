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
def ldnpgettext(self, domain: str, context: str, singular: str, plural: str, num: int) -> str | bytes:
    """Equivalent to ``dnpgettext()``, but the translation is returned in
        the preferred system encoding, if no other encoding was explicitly set
        with ``bind_textdomain_codeset()``.
        """
    return self._domains.get(domain, self).lnpgettext(context, singular, plural, num)