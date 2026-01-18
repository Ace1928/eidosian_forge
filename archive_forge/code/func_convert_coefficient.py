import re
import operator
from fractions import Fraction
import sys
def convert_coefficient(self, conversion_function):
    """
        Apply the specified conversion_function to the coefficient.
        e.g. monomial.convert_coefficient(float)
        """
    return Monomial(conversion_function(self._coefficient), self._vars)