import re
import operator
from fractions import Fraction
import sys
def split_variable(self, variable):
    """Split the specified variable from the others."""
    remaining_terms = {}
    exponent = 0
    for var, expo in self._vars:
        if var == variable:
            exponent = expo
        else:
            remaining_terms[var] = expo
    return (exponent, Monomial(self.get_coefficient(), remaining_terms))