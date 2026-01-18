from __future__ import annotations
import datetime
import decimal
import re
import warnings
from typing import TYPE_CHECKING, Any, cast, overload
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
def format_compact_decimal(number: float | decimal.Decimal | str, *, format_type: Literal['short', 'long']='short', locale: Locale | str | None=LC_NUMERIC, fraction_digits: int=0, numbering_system: Literal['default'] | str='latn') -> str:
    """Return the given decimal number formatted for a specific locale in compact form.

    >>> format_compact_decimal(12345, format_type="short", locale='en_US')
    u'12K'
    >>> format_compact_decimal(12345, format_type="long", locale='en_US')
    u'12 thousand'
    >>> format_compact_decimal(12345, format_type="short", locale='en_US', fraction_digits=2)
    u'12.34K'
    >>> format_compact_decimal(1234567, format_type="short", locale="ja_JP")
    u'123万'
    >>> format_compact_decimal(2345678, format_type="long", locale="mk")
    u'2 милиони'
    >>> format_compact_decimal(21000000, format_type="long", locale="mk")
    u'21 милион'
    >>> format_compact_decimal(12345, format_type="short", locale='ar_EG', fraction_digits=2, numbering_system='default')
    u'12٫34\xa0ألف'

    :param number: the number to format
    :param format_type: Compact format to use ("short" or "long")
    :param locale: the `Locale` object or locale identifier
    :param fraction_digits: Number of digits after the decimal point to use. Defaults to `0`.
    :param numbering_system: The numbering system used for formatting number symbols. Defaults to "latn".
                             The special value "default" will use the default numbering system of the locale.
    :raise `UnsupportedNumberingSystemError`: If the numbering system is not supported by the locale.
    """
    locale = Locale.parse(locale)
    compact_format = locale.compact_decimal_formats[format_type]
    number, format = _get_compact_format(number, compact_format, locale, fraction_digits)
    if format is None:
        format = locale.decimal_formats[None]
    pattern = parse_pattern(format)
    return pattern.apply(number, locale, decimal_quantization=False, numbering_system=numbering_system)