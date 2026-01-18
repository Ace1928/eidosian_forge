from __future__ import division
from future.utils import PYPY, PY26, bind_method
from decimal import Decimal, ROUND_HALF_EVEN
def newround(number, ndigits=None):
    """
    See Python 3 documentation: uses Banker's Rounding.

    Delegates to the __round__ method if for some reason this exists.

    If not, rounds a number to a given precision in decimal digits (default
    0 digits). This returns an int when called with one argument,
    otherwise the same type as the number. ndigits may be negative.

    See the test_round method in future/tests/test_builtins.py for
    examples.
    """
    return_int = False
    if ndigits is None:
        return_int = True
        ndigits = 0
    if hasattr(number, '__round__'):
        return number.__round__(ndigits)
    exponent = Decimal('10') ** (-ndigits)
    if 'numpy' in repr(type(number)):
        number = float(number)
    if isinstance(number, Decimal):
        d = number
    elif not PY26:
        d = Decimal.from_float(number)
    else:
        d = from_float_26(number)
    if ndigits < 0:
        result = newround(d / exponent) * exponent
    else:
        result = d.quantize(exponent, rounding=ROUND_HALF_EVEN)
    if return_int:
        return int(result)
    else:
        return float(result)