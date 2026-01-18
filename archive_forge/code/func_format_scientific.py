from __future__ import annotations
import datetime
import decimal
import re
import warnings
from typing import TYPE_CHECKING, Any, cast, overload
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
def format_scientific(number: float | decimal.Decimal | str, format: str | NumberPattern | None=None, locale: Locale | str | None=LC_NUMERIC, decimal_quantization: bool=True, *, numbering_system: Literal['default'] | str='latn') -> str:
    """Return value formatted in scientific notation for a specific locale.

    >>> format_scientific(10000, locale='en_US')
    u'1E4'
    >>> format_scientific(10000, locale='ar_EG', numbering_system='default')
    u'1اس4'

    The format pattern can also be specified explicitly:

    >>> format_scientific(1234567, u'##0.##E00', locale='en_US')
    u'1.23E06'

    By default the locale is allowed to truncate and round a high-precision
    number by forcing its format pattern onto the decimal part. You can bypass
    this behavior with the `decimal_quantization` parameter:

    >>> format_scientific(1234.9876, u'#.##E0', locale='en_US')
    u'1.23E3'
    >>> format_scientific(1234.9876, u'#.##E0', locale='en_US', decimal_quantization=False)
    u'1.2349876E3'

    :param number: the number to format
    :param format:
    :param locale: the `Locale` object or locale identifier
    :param decimal_quantization: Truncate and round high-precision numbers to
                                 the format pattern. Defaults to `True`.
    :param numbering_system: The numbering system used for formatting number symbols. Defaults to "latn".
                             The special value "default" will use the default numbering system of the locale.
    :raise `UnsupportedNumberingSystemError`: If the numbering system is not supported by the locale.
    """
    locale = Locale.parse(locale)
    if not format:
        format = locale.scientific_formats[None]
    pattern = parse_pattern(format)
    return pattern.apply(number, locale, decimal_quantization=decimal_quantization, numbering_system=numbering_system)