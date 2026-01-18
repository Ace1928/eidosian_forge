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
def orbit_transversal(self, alpha, pairs=False):
    """Computes a transversal for the orbit of ``alpha`` as a set.

        Explanation
        ===========

        For a permutation group `G`, a transversal for the orbit
        `Orb = \\{g(\\alpha) | g \\in G\\}` is a set
        `\\{g_\\beta | g_\\beta(\\alpha) = \\beta\\}` for `\\beta \\in Orb`.
        Note that there may be more than one possible transversal.
        If ``pairs`` is set to ``True``, it returns the list of pairs
        `(\\beta, g_\\beta)`. For a proof of correctness, see [1], p.79

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import DihedralGroup
        >>> G = DihedralGroup(6)
        >>> G.orbit_transversal(0)
        [(5), (0 1 2 3 4 5), (0 5)(1 4)(2 3), (0 2 4)(1 3 5), (5)(0 4)(1 3), (0 3)(1 4)(2 5)]

        See Also
        ========

        orbit

        """
    return _orbit_transversal(self._degree, self.generators, alpha, pairs)