import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def copy_sign(self, a, b):
    """Copies the second operand's sign to the first one.

        In detail, it returns a copy of the first operand with the sign
        equal to the sign of the second operand.

        >>> ExtendedContext.copy_sign(Decimal( '1.50'), Decimal('7.33'))
        Decimal('1.50')
        >>> ExtendedContext.copy_sign(Decimal('-1.50'), Decimal('7.33'))
        Decimal('1.50')
        >>> ExtendedContext.copy_sign(Decimal( '1.50'), Decimal('-7.33'))
        Decimal('-1.50')
        >>> ExtendedContext.copy_sign(Decimal('-1.50'), Decimal('-7.33'))
        Decimal('-1.50')
        >>> ExtendedContext.copy_sign(1, -2)
        Decimal('-1')
        >>> ExtendedContext.copy_sign(Decimal(1), -2)
        Decimal('-1')
        >>> ExtendedContext.copy_sign(1, Decimal(-2))
        Decimal('-1')
        """
    a = _convert_other(a, raiseit=True)
    return a.copy_sign(b)