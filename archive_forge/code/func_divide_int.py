import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def divide_int(self, a, b):
    """Divides two numbers and returns the integer part of the result.

        >>> ExtendedContext.divide_int(Decimal('2'), Decimal('3'))
        Decimal('0')
        >>> ExtendedContext.divide_int(Decimal('10'), Decimal('3'))
        Decimal('3')
        >>> ExtendedContext.divide_int(Decimal('1'), Decimal('0.3'))
        Decimal('3')
        >>> ExtendedContext.divide_int(10, 3)
        Decimal('3')
        >>> ExtendedContext.divide_int(Decimal(10), 3)
        Decimal('3')
        >>> ExtendedContext.divide_int(10, Decimal(3))
        Decimal('3')
        """
    a = _convert_other(a, raiseit=True)
    r = a.__floordiv__(b, context=self)
    if r is NotImplemented:
        raise TypeError('Unable to convert %s to Decimal' % b)
    else:
        return r