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
@property
def basic_stabilizers(self):
    """
        Return a chain of stabilizers relative to a base and strong generating
        set.

        Explanation
        ===========

        The ``i``-th basic stabilizer `G^{(i)}` relative to a base
        `(b_1, b_2, \\dots, b_k)` is `G_{b_1, b_2, \\dots, b_{i-1}}`. For more
        information, see [1], pp. 87-89.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import AlternatingGroup
        >>> A = AlternatingGroup(4)
        >>> A.schreier_sims()
        >>> A.base
        [0, 1]
        >>> for g in A.basic_stabilizers:
        ...     print(g)
        ...
        PermutationGroup([
            (3)(0 1 2),
            (1 2 3)])
        PermutationGroup([
            (1 2 3)])

        See Also
        ========

        base, strong_gens, basic_orbits, basic_transversals

        """
    if self._transversals == []:
        self.schreier_sims()
    strong_gens = self._strong_gens
    base = self._base
    if not base:
        return []
    strong_gens_distr = _distribute_gens_by_base(base, strong_gens)
    basic_stabilizers = []
    for gens in strong_gens_distr:
        basic_stabilizers.append(PermutationGroup(gens))
    return basic_stabilizers