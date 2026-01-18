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
def coset_rank(self, g):
    """rank using Schreier-Sims representation.

        Explanation
        ===========

        The coset rank of ``g`` is the ordering number in which
        it appears in the lexicographic listing according to the
        coset decomposition

        The ordering is the same as in G.generate(method='coset').
        If ``g`` does not belong to the group it returns None.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation(0, 1, 3, 7, 6, 4)(2, 5)
        >>> b = Permutation(0, 1, 3, 2)(4, 5, 7, 6)
        >>> G = PermutationGroup([a, b])
        >>> c = Permutation(7)(2, 4)(3, 5)
        >>> G.coset_rank(c)
        16
        >>> G.coset_unrank(16)
        (7)(2 4)(3 5)

        See Also
        ========

        coset_factor

        """
    factors = self.coset_factor(g, True)
    if not factors:
        return None
    rank = 0
    b = 1
    transversals = self._transversals
    base = self._base
    basic_orbits = self._basic_orbits
    for i in range(len(base)):
        k = factors[i]
        j = basic_orbits[i].index(k)
        rank += b * j
        b = b * len(transversals[i])
    return rank