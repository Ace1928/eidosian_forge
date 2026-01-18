import re
import operator
from fractions import Fraction
import sys
def substitute_monomial(monomial):
    vars = monomial.get_vars()
    new_vars = []
    for var, expo in vars:
        if var not in d:
            new_vars.append((var, expo))
    poly = Polynomial((Monomial(monomial.get_coefficient(), tuple(new_vars)),))
    for var, expo in vars:
        if var in d:
            poly = poly * d[var] ** expo
    return poly