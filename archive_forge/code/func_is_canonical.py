import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def is_canonical(self, a):
    """Return True if the operand is canonical; otherwise return False.

        Currently, the encoding of a Decimal instance is always
        canonical, so this method returns True for any Decimal.

        >>> ExtendedContext.is_canonical(Decimal('2.50'))
        True
        """
    if not isinstance(a, Decimal):
        raise TypeError('is_canonical requires a Decimal as an argument.')
    return a.is_canonical()