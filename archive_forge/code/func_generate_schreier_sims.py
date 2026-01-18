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
def generate_schreier_sims(self, af=False):
    """Yield group elements using the Schreier-Sims representation
        in coset_rank order

        If ``af = True`` it yields the array form of the permutations

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([0, 2, 1, 3])
        >>> b = Permutation([0, 2, 3, 1])
        >>> g = PermutationGroup([a, b])
        >>> list(g.generate_schreier_sims(af=True))
        [[0, 1, 2, 3], [0, 2, 1, 3], [0, 3, 2, 1],
         [0, 1, 3, 2], [0, 2, 3, 1], [0, 3, 1, 2]]
        """
    n = self._degree
    u = self.basic_transversals
    basic_orbits = self._basic_orbits
    if len(u) == 0:
        for x in self.generators:
            if af:
                yield x._array_form
            else:
                yield x
        return
    if len(u) == 1:
        for i in basic_orbits[0]:
            if af:
                yield u[0][i]._array_form
            else:
                yield u[0][i]
        return
    u = list(reversed(u))
    basic_orbits = basic_orbits[::-1]
    stg = [list(range(n))]
    posmax = [len(x) for x in u]
    n1 = len(posmax) - 1
    pos = [0] * n1
    h = 0
    while 1:
        if pos[h] >= posmax[h]:
            if h == 0:
                return
            pos[h] = 0
            h -= 1
            stg.pop()
            continue
        p = _af_rmul(u[h][basic_orbits[h][pos[h]]]._array_form, stg[-1])
        pos[h] += 1
        stg.append(p)
        h += 1
        if h == n1:
            if af:
                for i in basic_orbits[-1]:
                    p = _af_rmul(u[-1][i]._array_form, stg[-1])
                    yield p
            else:
                for i in basic_orbits[-1]:
                    p = _af_rmul(u[-1][i]._array_form, stg[-1])
                    p1 = _af_new(p)
                    yield p1
            stg.pop()
            h -= 1