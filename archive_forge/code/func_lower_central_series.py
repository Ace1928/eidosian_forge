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
def lower_central_series(self):
    """Return the lower central series for the group.

        The lower central series for a group `G` is the series
        `G = G_0 > G_1 > G_2 > \\ldots` where
        `G_k = [G, G_{k-1}]`, i.e. every term after the first is equal to the
        commutator of `G` and the previous term in `G1` ([1], p.29).

        Returns
        =======

        A list of permutation groups in the order `G = G_0, G_1, G_2, \\ldots`

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import (AlternatingGroup,
        ... DihedralGroup)
        >>> A = AlternatingGroup(4)
        >>> len(A.lower_central_series())
        2
        >>> A.lower_central_series()[1].is_subgroup(DihedralGroup(2))
        True

        See Also
        ========

        commutator, derived_series

        """
    res = [self]
    current = self
    nxt = self.commutator(self, current)
    while not current.is_subgroup(nxt):
        res.append(nxt)
        current = nxt
        nxt = self.commutator(self, current)
    return res