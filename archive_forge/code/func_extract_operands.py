from __future__ import annotations
import decimal
import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Callable
def extract_operands(source: float | decimal.Decimal) -> tuple[decimal.Decimal | int, int, int, int, int, int, Literal[0], Literal[0]]:
    """Extract operands from a decimal, a float or an int, according to `CLDR rules`_.

    The result is an 8-tuple (n, i, v, w, f, t, c, e), where those symbols are as follows:

    ====== ===============================================================
    Symbol Value
    ------ ---------------------------------------------------------------
    n      absolute value of the source number (integer and decimals).
    i      integer digits of n.
    v      number of visible fraction digits in n, with trailing zeros.
    w      number of visible fraction digits in n, without trailing zeros.
    f      visible fractional digits in n, with trailing zeros.
    t      visible fractional digits in n, without trailing zeros.
    c      compact decimal exponent value: exponent of the power of 10 used in compact decimal formatting.
    e      currently, synonym for ‘c’. however, may be redefined in the future.
    ====== ===============================================================

    .. _`CLDR rules`: https://www.unicode.org/reports/tr35/tr35-61/tr35-numbers.html#Operands

    :param source: A real number
    :type source: int|float|decimal.Decimal
    :return: A n-i-v-w-f-t-c-e tuple
    :rtype: tuple[decimal.Decimal, int, int, int, int, int, int, int]
    """
    n = abs(source)
    i = int(n)
    if isinstance(n, float):
        if i == n:
            n = i
        else:
            n = decimal.Decimal(str(n))
    if isinstance(n, decimal.Decimal):
        dec_tuple = n.as_tuple()
        exp = dec_tuple.exponent
        fraction_digits = dec_tuple.digits[exp:] if exp < 0 else ()
        trailing = ''.join((str(d) for d in fraction_digits))
        no_trailing = trailing.rstrip('0')
        v = len(trailing)
        w = len(no_trailing)
        f = int(trailing or 0)
        t = int(no_trailing or 0)
    else:
        v = w = f = t = 0
    c = e = 0
    return (n, i, v, w, f, t, c, e)