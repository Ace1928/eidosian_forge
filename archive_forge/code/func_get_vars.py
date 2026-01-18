import re
import operator
from fractions import Fraction
import sys
def get_vars(self):
    """
        Return a tuple of pairs (variable_name, exponent).
        """
    return self._vars