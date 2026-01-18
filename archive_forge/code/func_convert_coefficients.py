import re
import operator
from fractions import Fraction
import sys
def convert_coefficients(self, conversion_function):
    """Convert all coefficients using conversion_function."""
    return Polynomial(tuple([monomial.convert_coefficient(conversion_function) for monomial in self._monomials]))