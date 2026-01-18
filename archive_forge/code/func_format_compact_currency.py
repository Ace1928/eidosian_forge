from __future__ import annotations
import datetime
import decimal
import re
import warnings
from typing import TYPE_CHECKING, Any, cast, overload
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
def format_compact_currency(number: float | decimal.Decimal | str, currency: str, *, format_type: Literal['short']='short', locale: Locale | str | None=LC_NUMERIC, fraction_digits: int=0, numbering_system: Literal['default'] | str='latn') -> str:
    """Format a number as a currency value in compact form.

    >>> format_compact_currency(12345, 'USD', locale='en_US')
    u'$12K'
    >>> format_compact_currency(123456789, 'USD', locale='en_US', fraction_digits=2)
    u'$123.46M'
    >>> format_compact_currency(123456789, 'EUR', locale='de_DE', fraction_digits=1)
    '123,5\xa0Mio.\xa0€'

    :param number: the number to format
    :param currency: the currency code
    :param format_type: the compact format type to use. Defaults to "short".
    :param locale: the `Locale` object or locale identifier
    :param fraction_digits: Number of digits after the decimal point to use. Defaults to `0`.
    :param numbering_system: The numbering system used for formatting number symbols. Defaults to "latn".
                             The special value "default" will use the default numbering system of the locale.
    :raise `UnsupportedNumberingSystemError`: If the numbering system is not supported by the locale.
    """
    locale = Locale.parse(locale)
    try:
        compact_format = locale.compact_currency_formats[format_type]
    except KeyError as error:
        raise UnknownCurrencyFormatError(f'{format_type!r} is not a known compact currency format type') from error
    number, format = _get_compact_format(number, compact_format, locale, fraction_digits)
    if format is None or '¤' not in str(format):
        for magnitude in compact_format['other']:
            format = compact_format['other'][magnitude].pattern
            if '¤' not in format:
                continue
            format = re.sub('[^0\\s\\¤]', '', format)
            format = re.sub('(\\s)\\s+', '\\1', format).strip()
            break
    if format is None:
        raise ValueError('No compact currency format found for the given number and locale.')
    pattern = parse_pattern(format)
    return pattern.apply(number, locale, currency=currency, currency_digits=False, decimal_quantization=False, numbering_system=numbering_system)