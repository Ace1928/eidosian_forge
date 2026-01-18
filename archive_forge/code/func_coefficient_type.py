import re
import operator
from fractions import Fraction
import sys
def coefficient_type(self, the_type=int):
    """Returns the type of the coefficients."""
    for monomial in self._monomials:
        the_type = _storage_type_policy(the_type, monomial.coefficient_type())
    return the_type