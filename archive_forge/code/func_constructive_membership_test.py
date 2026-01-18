from sympy.ntheory.primetest import isprime
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.printing.defaults import DefaultPrinting
from sympy.combinatorics.free_groups import free_group
def constructive_membership_test(self, ipcgs, g):
    """
        Return the exponent vector for induced pcgs.
        """
    e = [0] * len(ipcgs)
    h = g
    d = self.depth(h)
    for i, gen in enumerate(ipcgs):
        while self.depth(gen) == d:
            f = self.leading_exponent(h) * self.leading_exponent(gen)
            f = f % self.relative_order[d - 1]
            h = gen ** (-f) * h
            e[i] = f
            d = self.depth(h)
    if h == 1:
        return e
    return False