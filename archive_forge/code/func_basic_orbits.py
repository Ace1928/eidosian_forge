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
def basic_orbits(self):
    """
        Return the basic orbits relative to a base and strong generating set.

        Explanation
        ===========

        If `(b_1, b_2, \\dots, b_k)` is a base for a group `G`, and
        `G^{(i)} = G_{b_1, b_2, \\dots, b_{i-1}}` is the ``i``-th basic stabilizer
        (so that `G^{(1)} = G`), the ``i``-th basic orbit relative to this base
        is the orbit of `b_i` under `G^{(i)}`. See [1], pp. 87-89 for more
        information.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> S = SymmetricGroup(4)
        >>> S.basic_orbits
        [[0, 1, 2, 3], [1, 2, 3], [2, 3]]

        See Also
        ========

        base, strong_gens, basic_transversals, basic_stabilizers

        """
    if self._basic_orbits == []:
        self.schreier_sims()
    return self._basic_orbits