import re
import warnings
from enum import Enum
from math import gcd
def linear_solve(self, symbol):
    """Return a, b such that a * symbol + b == self.

        If self is not linear with respect to symbol, raise RuntimeError.
        """
    b = self.substitute({symbol: as_number(0)})
    ax = self - b
    a = ax.substitute({symbol: as_number(1)})
    zero, _ = as_numer_denom(a * symbol - ax)
    if zero != as_number(0):
        raise RuntimeError(f'not a {symbol}-linear equation: {a} * {symbol} + {b} == {self}')
    return (a, b)