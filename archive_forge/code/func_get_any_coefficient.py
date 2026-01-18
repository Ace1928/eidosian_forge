import re
import operator
from fractions import Fraction
import sys
def get_any_coefficient(self):
    if len(self._monomials) == 0:
        return None
    else:
        return self._monomials[0].get_coefficient()