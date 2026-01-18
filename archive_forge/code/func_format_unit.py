from __future__ import annotations
import decimal
from typing import TYPE_CHECKING
from babel.core import Locale
from babel.numbers import LC_NUMERIC, format_decimal
def format_unit(value: str | float | decimal.Decimal, measurement_unit: str, length: Literal['short', 'long', 'narrow']='long', format: str | None=None, locale: Locale | str | None=LC_NUMERIC, *, numbering_system: Literal['default'] | str='latn') -> str:
    """Format a value of a given unit.

    Values are formatted according to the locale's usual pluralization rules
    and number formats.

    >>> format_unit(12, 'length-meter', locale='ro_RO')
    u'12 metri'
    >>> format_unit(15.5, 'length-mile', locale='fi_FI')
    u'15,5 mailia'
    >>> format_unit(1200, 'pressure-millimeter-ofhg', locale='nb')
    u'1\\xa0200 millimeter kvikks\\xf8lv'
    >>> format_unit(270, 'ton', locale='en')
    u'270 tons'
    >>> format_unit(1234.5, 'kilogram', locale='ar_EG', numbering_system='default')
    u'1٬234٫5 كيلوغرام'

    Number formats may be overridden with the ``format`` parameter.

    >>> import decimal
    >>> format_unit(decimal.Decimal("-42.774"), 'temperature-celsius', 'short', format='#.0', locale='fr')
    u'-42,8\\u202f\\xb0C'

    The locale's usual pluralization rules are respected.

    >>> format_unit(1, 'length-meter', locale='ro_RO')
    u'1 metru'
    >>> format_unit(0, 'length-mile', locale='cy')
    u'0 mi'
    >>> format_unit(1, 'length-mile', locale='cy')
    u'1 filltir'
    >>> format_unit(3, 'length-mile', locale='cy')
    u'3 milltir'

    >>> format_unit(15, 'length-horse', locale='fi')
    Traceback (most recent call last):
        ...
    UnknownUnitError: length-horse is not a known unit in fi

    .. versionadded:: 2.2.0

    :param value: the value to format. If this is a string, no number formatting will be attempted.
    :param measurement_unit: the code of a measurement unit.
                             Known units can be found in the CLDR Unit Validity XML file:
                             https://unicode.org/repos/cldr/tags/latest/common/validity/unit.xml
    :param length: "short", "long" or "narrow"
    :param format: An optional format, as accepted by `format_decimal`.
    :param locale: the `Locale` object or locale identifier
    :param numbering_system: The numbering system used for formatting number symbols. Defaults to "latn".
                             The special value "default" will use the default numbering system of the locale.
    :raise `UnsupportedNumberingSystemError`: If the numbering system is not supported by the locale.
    """
    locale = Locale.parse(locale)
    q_unit = _find_unit_pattern(measurement_unit, locale=locale)
    if not q_unit:
        raise UnknownUnitError(unit=measurement_unit, locale=locale)
    unit_patterns = locale._data['unit_patterns'][q_unit].get(length, {})
    if isinstance(value, str):
        formatted_value = value
        plural_form = 'one'
    else:
        formatted_value = format_decimal(value, format, locale, numbering_system=numbering_system)
        plural_form = locale.plural_form(value)
    if plural_form in unit_patterns:
        return unit_patterns[plural_form].format(formatted_value)
    fallback_name = get_unit_name(measurement_unit, length=length, locale=locale)
    return f'{formatted_value} {fallback_name or measurement_unit}'