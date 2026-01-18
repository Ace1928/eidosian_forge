from __future__ import annotations
import datetime
import decimal
import re
import warnings
from typing import TYPE_CHECKING, Any, cast, overload
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
def scientific_notation_elements(self, value: decimal.Decimal, locale: Locale | str | None, *, numbering_system: Literal['default'] | str='latn') -> tuple[decimal.Decimal, int, str]:
    """ Returns normalized scientific notation components of a value.
        """
    exp = value.adjusted()
    value = value * get_decimal_quantum(exp)
    assert value.adjusted() == 0
    lead_shift = max([1, min(self.int_prec)]) - 1
    exp = exp - lead_shift
    value = value * get_decimal_quantum(-lead_shift)
    exp_sign = ''
    if exp < 0:
        exp_sign = get_minus_sign_symbol(locale, numbering_system=numbering_system)
    elif self.exp_plus:
        exp_sign = get_plus_sign_symbol(locale, numbering_system=numbering_system)
    exp = abs(exp)
    return (value, exp, exp_sign)