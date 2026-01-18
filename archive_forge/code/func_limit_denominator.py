from decimal import Decimal
import math
import numbers
import operator
import re
import sys
def limit_denominator(self, max_denominator=1000000):
    """Closest Fraction to self with denominator at most max_denominator.

        >>> Fraction('3.141592653589793').limit_denominator(10)
        Fraction(22, 7)
        >>> Fraction('3.141592653589793').limit_denominator(100)
        Fraction(311, 99)
        >>> Fraction(4321, 8765).limit_denominator(10000)
        Fraction(4321, 8765)

        """
    if max_denominator < 1:
        raise ValueError('max_denominator should be at least 1')
    if self._denominator <= max_denominator:
        return Fraction(self)
    p0, q0, p1, q1 = (0, 1, 1, 0)
    n, d = (self._numerator, self._denominator)
    while True:
        a = n // d
        q2 = q0 + a * q1
        if q2 > max_denominator:
            break
        p0, q0, p1, q1 = (p1, q1, p0 + a * p1, q2)
        n, d = (d, n - a * d)
    k = (max_denominator - q0) // q1
    bound1 = Fraction(p0 + k * p1, q0 + k * q1)
    bound2 = Fraction(p1, q1)
    if abs(bound2 - self) <= abs(bound1 - self):
        return bound2
    else:
        return bound1