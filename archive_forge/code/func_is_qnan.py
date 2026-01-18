import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def is_qnan(self, a):
    """Return True if the operand is a quiet NaN; otherwise return False.

        >>> ExtendedContext.is_qnan(Decimal('2.50'))
        False
        >>> ExtendedContext.is_qnan(Decimal('NaN'))
        True
        >>> ExtendedContext.is_qnan(Decimal('sNaN'))
        False
        >>> ExtendedContext.is_qnan(1)
        False
        """
    a = _convert_other(a, raiseit=True)
    return a.is_qnan()