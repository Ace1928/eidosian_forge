import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def getdigits(self, p):
    """Given an integer p >= 0, return floor(10**p)*log(10).

        For example, self.getdigits(3) returns 2302.
        """
    if p < 0:
        raise ValueError('p should be nonnegative')
    if p >= len(self.digits):
        extra = 3
        while True:
            M = 10 ** (p + extra + 2)
            digits = str(_div_nearest(_ilog(10 * M, M), 100))
            if digits[-extra:] != '0' * extra:
                break
            extra += 3
        self.digits = digits.rstrip('0')[:-1]
    return int(self.digits[:p + 1])