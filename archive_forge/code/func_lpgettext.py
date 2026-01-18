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
def lpgettext(self, context: str, message: str) -> str | bytes | object:
    """Equivalent to ``pgettext()``, but the translation is returned in the
        preferred system encoding, if no other encoding was explicitly set with
        ``bind_textdomain_codeset()``.
        """
    import warnings
    warnings.warn('lpgettext() is deprecated, use pgettext() instead', DeprecationWarning, stacklevel=2)
    tmsg = self.pgettext(context, message)
    encoding = getattr(self, '_output_charset', None) or locale.getpreferredencoding()
    return tmsg.encode(encoding) if isinstance(tmsg, str) else tmsg