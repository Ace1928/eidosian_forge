import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def is_normal(self, a):
    """Return True if the operand is a normal number;
        otherwise return False.

        >>> c = ExtendedContext.copy()
        >>> c.Emin = -999
        >>> c.Emax = 999
        >>> c.is_normal(Decimal('2.50'))
        True
        >>> c.is_normal(Decimal('0.1E-999'))
        False
        >>> c.is_normal(Decimal('0.00'))
        False
        >>> c.is_normal(Decimal('-Inf'))
        False
        >>> c.is_normal(Decimal('NaN'))
        False
        >>> c.is_normal(1)
        True
        """
    a = _convert_other(a, raiseit=True)
    return a.is_normal(context=self)