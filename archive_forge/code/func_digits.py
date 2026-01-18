from collections import defaultdict
from sympy.utilities.iterables import multiset, is_palindromic as _palindromic
from sympy.utilities.misc import as_int
def digits(n, b=10, digits=None):
    """
    Return a list of the digits of ``n`` in base ``b``. The first
    element in the list is ``b`` (or ``-b`` if ``n`` is negative).

    Examples
    ========

    >>> from sympy.ntheory.digits import digits
    >>> digits(35)
    [10, 3, 5]

    If the number is negative, the negative sign will be placed on the
    base (which is the first element in the returned list):

    >>> digits(-35)
    [-10, 3, 5]

    Bases other than 10 (and greater than 1) can be selected with ``b``:

    >>> digits(27, b=2)
    [2, 1, 1, 0, 1, 1]

    Use the ``digits`` keyword if a certain number of digits is desired:

    >>> digits(35, digits=4)
    [10, 0, 0, 3, 5]

    Parameters
    ==========

    n: integer
        The number whose digits are returned.

    b: integer
        The base in which digits are computed.

    digits: integer (or None for all digits)
        The number of digits to be returned (padded with zeros, if
        necessary).

    """
    b = as_int(b)
    n = as_int(n)
    if b < 2:
        raise ValueError('b must be greater than 1')
    else:
        x, y = (abs(n), [])
        while x >= b:
            x, r = divmod(x, b)
            y.append(r)
        y.append(x)
        y.append(-b if n < 0 else b)
        y.reverse()
        ndig = len(y) - 1
        if digits is not None:
            if ndig > digits:
                raise ValueError('For %s, at least %s digits are needed.' % (n, ndig))
            elif ndig < digits:
                y[1:1] = [0] * (digits - ndig)
        return y