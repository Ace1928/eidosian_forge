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
def coset_transversal(self, H):
    """Return a transversal of the right cosets of self by its subgroup H
        using the second method described in [1], Subsection 4.6.7

        """
    if not H.is_subgroup(self):
        raise ValueError('The argument must be a subgroup')
    if H.order() == 1:
        return self._elements
    self._schreier_sims(base=H.base)
    base = self.base
    base_ordering = _base_ordering(base, self.degree)
    identity = Permutation(self.degree - 1)
    transversals = self.basic_transversals[:]
    for l, t in enumerate(transversals):
        transversals[l] = sorted(t.values(), key=lambda x: base_ordering[base[l] ^ x])
    orbits = H.basic_orbits
    h_stabs = H.basic_stabilizers
    g_stabs = self.basic_stabilizers
    indices = [x.order() // y.order() for x, y in zip(g_stabs, h_stabs)]
    if len(g_stabs) > len(h_stabs):
        T = g_stabs[len(h_stabs)]._elements
    else:
        T = [identity]
    l = len(h_stabs) - 1
    t_len = len(T)
    while l > -1:
        T_next = []
        for u in transversals[l]:
            if u == identity:
                continue
            b = base_ordering[base[l] ^ u]
            for t in T:
                p = t * u
                if all((base_ordering[h ^ p] >= b for h in orbits[l])):
                    T_next.append(p)
                if t_len + len(T_next) == indices[l]:
                    break
            if t_len + len(T_next) == indices[l]:
                break
        T += T_next
        t_len += len(T_next)
        l -= 1
    T.remove(identity)
    T = [identity] + T
    return T