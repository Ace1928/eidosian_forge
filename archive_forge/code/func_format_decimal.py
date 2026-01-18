from __future__ import annotations
import datetime
import decimal
import re
import warnings
from typing import TYPE_CHECKING, Any, cast, overload
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
def format_decimal(number: float | decimal.Decimal | str, format: str | NumberPattern | None=None, locale: Locale | str | None=LC_NUMERIC, decimal_quantization: bool=True, group_separator: bool=True, *, numbering_system: Literal['default'] | str='latn') -> str:
    """Return the given decimal number formatted for a specific locale.

    >>> format_decimal(1.2345, locale='en_US')
    u'1.234'
    >>> format_decimal(1.2346, locale='en_US')
    u'1.235'
    >>> format_decimal(-1.2346, locale='en_US')
    u'-1.235'
    >>> format_decimal(1.2345, locale='sv_SE')
    u'1,234'
    >>> format_decimal(1.2345, locale='de')
    u'1,234'
    >>> format_decimal(1.2345, locale='ar_EG', numbering_system='default')
    u'1٫234'
    >>> format_decimal(1.2345, locale='ar_EG', numbering_system='latn')
    u'1.234'

    The appropriate thousands grouping and the decimal separator are used for
    each locale:

    >>> format_decimal(12345.5, locale='en_US')
    u'12,345.5'

    By default the locale is allowed to truncate and round a high-precision
    number by forcing its format pattern onto the decimal part. You can bypass
    this behavior with the `decimal_quantization` parameter:

    >>> format_decimal(1.2346, locale='en_US')
    u'1.235'
    >>> format_decimal(1.2346, locale='en_US', decimal_quantization=False)
    u'1.2346'
    >>> format_decimal(12345.67, locale='fr_CA', group_separator=False)
    u'12345,67'
    >>> format_decimal(12345.67, locale='en_US', group_separator=True)
    u'12,345.67'

    :param number: the number to format
    :param format:
    :param locale: the `Locale` object or locale identifier
    :param decimal_quantization: Truncate and round high-precision numbers to
                                 the format pattern. Defaults to `True`.
    :param group_separator: Boolean to switch group separator on/off in a locale's
                            number format.
    :param numbering_system: The numbering system used for formatting number symbols. Defaults to "latn".
                             The special value "default" will use the default numbering system of the locale.
    :raise `UnsupportedNumberingSystemError`: If the numbering system is not supported by the locale.
    """
    locale = Locale.parse(locale)
    if format is None:
        format = locale.decimal_formats[format]
    pattern = parse_pattern(format)
    return pattern.apply(number, locale, decimal_quantization=decimal_quantization, group_separator=group_separator, numbering_system=numbering_system)