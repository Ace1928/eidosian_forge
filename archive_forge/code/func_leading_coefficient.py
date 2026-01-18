import re
import operator
from fractions import Fraction
import sys
def leading_coefficient(self):
    """
        Assert univariance; return the leading coefficient.
        """
    assert self.is_univariate()
    if self._monomials:
        return self._monomials[-1].get_coefficient()
    else:
        return 0