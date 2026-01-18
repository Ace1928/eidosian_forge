import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def radix(self):
    """Just returns 10, as this is Decimal, :)

        >>> ExtendedContext.radix()
        Decimal('10')
        """
    return Decimal(10)