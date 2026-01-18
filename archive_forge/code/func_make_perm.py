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
def make_perm(self, n, seed=None):
    """
        Multiply ``n`` randomly selected permutations from
        pgroup together, starting with the identity
        permutation. If ``n`` is a list of integers, those
        integers will be used to select the permutations and they
        will be applied in L to R order: make_perm((A, B, C)) will
        give CBA(I) where I is the identity permutation.

        ``seed`` is used to set the seed for the random selection
        of permutations from pgroup. If this is a list of integers,
        the corresponding permutations from pgroup will be selected
        in the order give. This is mainly used for testing purposes.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a, b = [Permutation([1, 0, 3, 2]), Permutation([1, 3, 0, 2])]
        >>> G = PermutationGroup([a, b])
        >>> G.make_perm(1, [0])
        (0 1)(2 3)
        >>> G.make_perm(3, [0, 1, 0])
        (0 2 3 1)
        >>> G.make_perm([0, 1, 0])
        (0 2 3 1)

        See Also
        ========

        random
        """
    if is_sequence(n):
        if seed is not None:
            raise ValueError('If n is a sequence, seed should be None')
        n, seed = (len(n), n)
    else:
        try:
            n = int(n)
        except TypeError:
            raise ValueError('n must be an integer or a sequence.')
    randomrange = _randrange(seed)
    result = Permutation(list(range(self.degree)))
    m = len(self)
    for _ in range(n):
        p = self[randomrange(m)]
        result = rmul(result, p)
    return result