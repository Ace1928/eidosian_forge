import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def compare_total(self, a, b):
    """Compares two operands using their abstract representation.

        This is not like the standard compare, which use their numerical
        value. Note that a total ordering is defined for all possible abstract
        representations.

        >>> ExtendedContext.compare_total(Decimal('12.73'), Decimal('127.9'))
        Decimal('-1')
        >>> ExtendedContext.compare_total(Decimal('-127'),  Decimal('12'))
        Decimal('-1')
        >>> ExtendedContext.compare_total(Decimal('12.30'), Decimal('12.3'))
        Decimal('-1')
        >>> ExtendedContext.compare_total(Decimal('12.30'), Decimal('12.30'))
        Decimal('0')
        >>> ExtendedContext.compare_total(Decimal('12.3'),  Decimal('12.300'))
        Decimal('1')
        >>> ExtendedContext.compare_total(Decimal('12.3'),  Decimal('NaN'))
        Decimal('-1')
        >>> ExtendedContext.compare_total(1, 2)
        Decimal('-1')
        >>> ExtendedContext.compare_total(Decimal(1), 2)
        Decimal('-1')
        >>> ExtendedContext.compare_total(1, Decimal(2))
        Decimal('-1')
        """
    a = _convert_other(a, raiseit=True)
    return a.compare_total(b)