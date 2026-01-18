from math import factorial as _factorial, log, prod
from itertools import chain, islice, product
from sympy.combinatorics import Permutation
from sympy.combinatorics.permutations import (_af_commutes_with, _af_invert,
from sympy.combinatorics.util import (_check_cycles_alt_sym,
from sympy.core import Basic
from sympy.core.random import _randrange, randrange, choice
from sympy.core.symbol import Symbol
from sympy.core.sympify import _sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.ntheory import primefactors, sieve
from sympy.ntheory.factor_ import (factorint, multiplicity)
from sympy.ntheory.primetest import isprime
from sympy.utilities.iterables import has_variety, is_sequence, uniq
def generator_product(self, g, original=False):
    """
        Return a list of strong generators `[s1, \\dots, sn]`
        s.t `g = sn \\times \\dots \\times s1`. If ``original=True``, make the
        list contain only the original group generators

        """
    product = []
    if g.is_identity:
        return []
    if g in self.strong_gens:
        if not original or g in self.generators:
            return [g]
        else:
            slp = self._strong_gens_slp[g]
            for s in slp:
                product.extend(self.generator_product(s, original=True))
            return product
    elif g ** (-1) in self.strong_gens:
        g = g ** (-1)
        if not original or g in self.generators:
            return [g ** (-1)]
        else:
            slp = self._strong_gens_slp[g]
            for s in slp:
                product.extend(self.generator_product(s, original=True))
            l = len(product)
            product = [product[l - i - 1] ** (-1) for i in range(l)]
            return product
    f = self.coset_factor(g, True)
    for i, j in enumerate(f):
        slp = self._transversal_slp[i][j]
        for s in slp:
            if not original:
                product.append(self.strong_gens[s])
            else:
                s = self.strong_gens[s]
                product.extend(self.generator_product(s, original=True))
    return product