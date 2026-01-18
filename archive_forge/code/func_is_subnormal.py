import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def is_subnormal(self, a):
    """Return True if the operand is subnormal; otherwise return False.

        >>> c = ExtendedContext.copy()
        >>> c.Emin = -999
        >>> c.Emax = 999
        >>> c.is_subnormal(Decimal('2.50'))
        False
        >>> c.is_subnormal(Decimal('0.1E-999'))
        True
        >>> c.is_subnormal(Decimal('0.00'))
        False
        >>> c.is_subnormal(Decimal('-Inf'))
        False
        >>> c.is_subnormal(Decimal('NaN'))
        False
        >>> c.is_subnormal(1)
        False
        """
    a = _convert_other(a, raiseit=True)
    return a.is_subnormal(context=self)