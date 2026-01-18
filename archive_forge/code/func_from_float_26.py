from __future__ import division
from future.utils import PYPY, PY26, bind_method
from decimal import Decimal, ROUND_HALF_EVEN
def from_float_26(f):
    """Converts a float to a decimal number, exactly.

    Note that Decimal.from_float(0.1) is not the same as Decimal('0.1').
    Since 0.1 is not exactly representable in binary floating point, the
    value is stored as the nearest representable value which is
    0x1.999999999999ap-4.  The exact equivalent of the value in decimal
    is 0.1000000000000000055511151231257827021181583404541015625.

    >>> Decimal.from_float(0.1)
    Decimal('0.1000000000000000055511151231257827021181583404541015625')
    >>> Decimal.from_float(float('nan'))
    Decimal('NaN')
    >>> Decimal.from_float(float('inf'))
    Decimal('Infinity')
    >>> Decimal.from_float(-float('inf'))
    Decimal('-Infinity')
    >>> Decimal.from_float(-0.0)
    Decimal('-0')

    """
    import math as _math
    from decimal import _dec_from_triple
    if isinstance(f, (int, long)):
        return Decimal(f)
    if _math.isinf(f) or _math.isnan(f):
        return Decimal(repr(f))
    if _math.copysign(1.0, f) == 1.0:
        sign = 0
    else:
        sign = 1
    n, d = abs(f).as_integer_ratio()

    def bit_length(d):
        if d != 0:
            return len(bin(abs(d))) - 2
        else:
            return 0
    k = bit_length(d) - 1
    result = _dec_from_triple(sign, str(n * 5 ** k), -k)
    return result