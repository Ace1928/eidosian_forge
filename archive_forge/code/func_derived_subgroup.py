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
def derived_subgroup(self):
    """Compute the derived subgroup.

        Explanation
        ===========

        The derived subgroup, or commutator subgroup is the subgroup generated
        by all commutators `[g, h] = hgh^{-1}g^{-1}` for `g, h\\in G` ; it is
        equal to the normal closure of the set of commutators of the generators
        ([1], p.28, [11]).

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([1, 0, 2, 4, 3])
        >>> b = Permutation([0, 1, 3, 2, 4])
        >>> G = PermutationGroup([a, b])
        >>> C = G.derived_subgroup()
        >>> list(C.generate(af=True))
        [[0, 1, 2, 3, 4], [0, 1, 3, 4, 2], [0, 1, 4, 2, 3]]

        See Also
        ========

        derived_series

        """
    r = self._r
    gens = [p._array_form for p in self.generators]
    set_commutators = set()
    degree = self._degree
    rng = list(range(degree))
    for i in range(r):
        for j in range(r):
            p1 = gens[i]
            p2 = gens[j]
            c = list(range(degree))
            for k in rng:
                c[p2[p1[k]]] = p1[p2[k]]
            ct = tuple(c)
            if ct not in set_commutators:
                set_commutators.add(ct)
    cms = [_af_new(p) for p in set_commutators]
    G2 = self.normal_closure(cms)
    return G2