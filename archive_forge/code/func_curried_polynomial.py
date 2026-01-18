import re
import operator
from fractions import Fraction
import sys
def curried_polynomial(self, variable):
    """
        Return a polynomial in the variable whose coefficients are polynomials in
        the other variables.
        """
    poly = Polynomial()
    for monomial in self._monomials:
        exponent, remainder = monomial.split_variable(variable)
        poly = poly + (Polynomial.from_variable_name(variable) ** exponent).convert_coefficients(Polynomial.constant_polynomial) * Polynomial.constant_polynomial(Polynomial((remainder,)))
    return poly