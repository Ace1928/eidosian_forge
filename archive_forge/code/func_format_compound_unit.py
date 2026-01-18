from __future__ import annotations
import decimal
from typing import TYPE_CHECKING
from babel.core import Locale
from babel.numbers import LC_NUMERIC, format_decimal
def format_compound_unit(numerator_value: str | float | decimal.Decimal, numerator_unit: str | None=None, denominator_value: str | float | decimal.Decimal=1, denominator_unit: str | None=None, length: Literal['short', 'long', 'narrow']='long', format: str | None=None, locale: Locale | str | None=LC_NUMERIC, *, numbering_system: Literal['default'] | str='latn') -> str | None:
    """
    Format a compound number value, i.e. "kilometers per hour" or similar.

    Both unit specifiers are optional to allow for formatting of arbitrary values still according
    to the locale's general "per" formatting specifier.

    >>> format_compound_unit(7, denominator_value=11, length="short", locale="pt")
    '7/11'

    >>> format_compound_unit(150, "kilometer", denominator_unit="hour", locale="sv")
    '150 kilometer per timme'

    >>> format_compound_unit(150, "kilowatt", denominator_unit="year", locale="fi")
    '150 kilowattia / vuosi'

    >>> format_compound_unit(32.5, "ton", 15, denominator_unit="hour", locale="en")
    '32.5 tons per 15 hours'

    >>> format_compound_unit(1234.5, "ton", 15, denominator_unit="hour", locale="ar_EG", numbering_system="arab")
    '1٬234٫5 طن لكل 15 ساعة'

    >>> format_compound_unit(160, denominator_unit="square-meter", locale="fr")
    '160 par m\\xe8tre carr\\xe9'

    >>> format_compound_unit(4, "meter", "ratakisko", length="short", locale="fi")
    '4 m/ratakisko'

    >>> format_compound_unit(35, "minute", denominator_unit="fathom", locale="sv")
    '35 minuter per famn'

    >>> from babel.numbers import format_currency
    >>> format_compound_unit(format_currency(35, "JPY", locale="de"), denominator_unit="liter", locale="de")
    '35\\xa0\\xa5 pro Liter'

    See https://www.unicode.org/reports/tr35/tr35-general.html#perUnitPatterns

    :param numerator_value: The numerator value. This may be a string,
                            in which case it is considered preformatted and the unit is ignored.
    :param numerator_unit: The numerator unit. See `format_unit`.
    :param denominator_value: The denominator value. This may be a string,
                              in which case it is considered preformatted and the unit is ignored.
    :param denominator_unit: The denominator unit. See `format_unit`.
    :param length: The formatting length. "short", "long" or "narrow"
    :param format: An optional format, as accepted by `format_decimal`.
    :param locale: the `Locale` object or locale identifier
    :param numbering_system: The numbering system used for formatting number symbols. Defaults to "latn".
                             The special value "default" will use the default numbering system of the locale.
    :return: A formatted compound value.
    :raise `UnsupportedNumberingSystemError`: If the numbering system is not supported by the locale.
    """
    locale = Locale.parse(locale)
    if numerator_unit and denominator_unit and (denominator_value == 1):
        compound_unit = _find_compound_unit(numerator_unit, denominator_unit, locale=locale)
        if compound_unit:
            return format_unit(numerator_value, compound_unit, length=length, format=format, locale=locale, numbering_system=numbering_system)
    if isinstance(numerator_value, str):
        formatted_numerator = numerator_value
    elif numerator_unit:
        formatted_numerator = format_unit(numerator_value, numerator_unit, length=length, format=format, locale=locale, numbering_system=numbering_system)
    else:
        formatted_numerator = format_decimal(numerator_value, format=format, locale=locale, numbering_system=numbering_system)
    if isinstance(denominator_value, str):
        formatted_denominator = denominator_value
    elif denominator_unit:
        if denominator_value == 1:
            denominator_unit = _find_unit_pattern(denominator_unit, locale=locale)
            per_pattern = locale._data['unit_patterns'].get(denominator_unit, {}).get(length, {}).get('per')
            if per_pattern:
                return per_pattern.format(formatted_numerator)
            denominator_value = ''
        formatted_denominator = format_unit(denominator_value, measurement_unit=denominator_unit or '', length=length, format=format, locale=locale, numbering_system=numbering_system).strip()
    else:
        formatted_denominator = format_decimal(denominator_value, format=format, locale=locale, numbering_system=numbering_system)
    per_pattern = locale._data['compound_unit_patterns'].get('per', {}).get(length, {}).get('compound', '{0}/{1}')
    return per_pattern.format(formatted_numerator, formatted_denominator)