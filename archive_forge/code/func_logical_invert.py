import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def logical_invert(self, a):
    """Invert all the digits in the operand.

        The operand must be a logical number.

        >>> ExtendedContext.logical_invert(Decimal('0'))
        Decimal('111111111')
        >>> ExtendedContext.logical_invert(Decimal('1'))
        Decimal('111111110')
        >>> ExtendedContext.logical_invert(Decimal('111111111'))
        Decimal('0')
        >>> ExtendedContext.logical_invert(Decimal('101010101'))
        Decimal('10101010')
        >>> ExtendedContext.logical_invert(1101)
        Decimal('111110010')
        """
    a = _convert_other(a, raiseit=True)
    return a.logical_invert(context=self)