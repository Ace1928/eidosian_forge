from decimal import Decimal
import math
import numbers
import operator
import re
import sys
@classmethod
def from_decimal(cls, dec):
    """Converts a finite Decimal instance to a rational number, exactly."""
    from decimal import Decimal
    if isinstance(dec, numbers.Integral):
        dec = Decimal(int(dec))
    elif not isinstance(dec, Decimal):
        raise TypeError('%s.from_decimal() only takes Decimals, not %r (%s)' % (cls.__name__, dec, type(dec).__name__))
    return cls(*dec.as_integer_ratio())