from __future__ import annotations
import datetime
import decimal
import re
import warnings
from typing import TYPE_CHECKING, Any, cast, overload
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
def get_minus_sign_symbol(locale: Locale | str | None=LC_NUMERIC, *, numbering_system: Literal['default'] | str='latn') -> str:
    """Return the plus sign symbol used by the current locale.

    >>> get_minus_sign_symbol('en_US')
    u'-'
    >>> get_minus_sign_symbol('ar_EG', numbering_system='default')
    u'\u061c-'
    >>> get_minus_sign_symbol('ar_EG', numbering_system='latn')
    u'\u200e-'

    :param locale: the `Locale` object or locale identifier
    :param numbering_system: The numbering system used for fetching the symbol. Defaults to "latn".
                             The special value "default" will use the default numbering system of the locale.
    :raise `UnsupportedNumberingSystemError`: if the numbering system is not supported by the locale.
    """
    return _get_number_symbols(locale, numbering_system=numbering_system).get('minusSign', '-')