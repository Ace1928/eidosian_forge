import re
import operator
from fractions import Fraction
import sys
def factor_out_variables(self):
    if self._monomials == ():
        return self

    def intersect(lists):
        s = set(lists[0])
        for l in lists:
            s &= set(l)
        return s
    non_trivial_variables = intersect([monomial.variables() for monomial in self._monomials])
    lowest_powers = dict([(var, 1000000) for var in non_trivial_variables])

    def safe_dict(d, var):
        if var in d:
            return d[var]
        else:
            return 0
    for monomial in self._monomials:
        for var, expo in monomial.get_vars():
            lowest_powers[var] = min(safe_dict(lowest_powers, var), expo)
    return Polynomial(tuple([monomial.reduce_exponents(lowest_powers) for monomial in self._monomials]))