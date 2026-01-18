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
def max_div(self):
    """Maximum proper divisor of the degree of a permutation group.

        Explanation
        ===========

        Obviously, this is the degree divided by its minimal proper divisor
        (larger than ``1``, if one exists). As it is guaranteed to be prime,
        the ``sieve`` from ``sympy.ntheory`` is used.
        This function is also used as an optimization tool for the functions
        ``minimal_block`` and ``_union_find_merge``.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> G = PermutationGroup([Permutation([0, 2, 1, 3])])
        >>> G.max_div
        2

        See Also
        ========

        minimal_block, _union_find_merge

        """
    if self._max_div is not None:
        return self._max_div
    n = self.degree
    if n == 1:
        return 1
    for x in sieve:
        if n % x == 0:
            d = n // x
            self._max_div = d
            return d