import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def copy_abs(self, a):
    """Returns a copy of the operand with the sign set to 0.

        >>> ExtendedContext.copy_abs(Decimal('2.1'))
        Decimal('2.1')
        >>> ExtendedContext.copy_abs(Decimal('-100'))
        Decimal('100')
        >>> ExtendedContext.copy_abs(-1)
        Decimal('1')
        """
    a = _convert_other(a, raiseit=True)
    return a.copy_abs()