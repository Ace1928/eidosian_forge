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
def lnpgettext(self, context: str, singular: str, plural: str, num: int) -> str | bytes:
    """Equivalent to ``npgettext()``, but the translation is returned in the
        preferred system encoding, if no other encoding was explicitly set with
        ``bind_textdomain_codeset()``.
        """
    import warnings
    warnings.warn('lnpgettext() is deprecated, use npgettext() instead', DeprecationWarning, stacklevel=2)
    ctxt_msg_id = self.CONTEXT_ENCODING % (context, singular)
    try:
        tmsg = self._catalog[ctxt_msg_id, self.plural(num)]
        encoding = getattr(self, '_output_charset', None) or locale.getpreferredencoding()
        return tmsg.encode(encoding)
    except KeyError:
        if self._fallback:
            return self._fallback.lnpgettext(context, singular, plural, num)
        if num == 1:
            return singular
        else:
            return plural