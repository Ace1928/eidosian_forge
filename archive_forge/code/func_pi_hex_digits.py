import math
from sympy.utilities.misc import as_int
def pi_hex_digits(n, prec=14):
    """Returns a string containing ``prec`` (default 14) digits
    starting at the nth digit of pi in hex. Counting of digits
    starts at 0 and the decimal is not counted, so for n = 0 the
    returned value starts with 3; n = 1 corresponds to the first
    digit past the decimal point (which in hex is 2).

    Examples
    ========

    >>> from sympy.ntheory.bbp_pi import pi_hex_digits
    >>> pi_hex_digits(0)
    '3243f6a8885a30'
    >>> pi_hex_digits(0, 3)
    '324'

    References
    ==========

    .. [1] http://www.numberworld.org/digits/Pi/
    """
    n, prec = (as_int(n), as_int(prec))
    if n < 0:
        raise ValueError('n cannot be negative')
    if prec == 0:
        return ''
    n -= 1
    a = [4, 2, 1, 1]
    j = [1, 4, 5, 6]
    D = _dn(n, prec)
    x = +(a[0] * _series(j[0], n, prec) - a[1] * _series(j[1], n, prec) - a[2] * _series(j[2], n, prec) - a[3] * _series(j[3], n, prec)) & 16 ** D - 1
    s = ('%0' + '%ix' % prec) % (x // 16 ** (D - prec))
    return s