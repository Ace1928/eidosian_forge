import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def create_decimal(self, num='0'):
    """Creates a new Decimal instance but using self as context.

        This method implements the to-number operation of the
        IBM Decimal specification."""
    if isinstance(num, str) and (num != num.strip() or '_' in num):
        return self._raise_error(ConversionSyntax, 'trailing or leading whitespace and underscores are not permitted.')
    d = Decimal(num, context=self)
    if d._isnan() and len(d._int) > self.prec - self.clamp:
        return self._raise_error(ConversionSyntax, 'diagnostic info too long in NaN')
    return d._fix(self)