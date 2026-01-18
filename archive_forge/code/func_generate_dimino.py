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
def generate_dimino(self, af=False):
    """Yield group elements using Dimino's algorithm.

        If ``af == True`` it yields the array form of the permutations.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([0, 2, 1, 3])
        >>> b = Permutation([0, 2, 3, 1])
        >>> g = PermutationGroup([a, b])
        >>> list(g.generate_dimino(af=True))
        [[0, 1, 2, 3], [0, 2, 1, 3], [0, 2, 3, 1],
         [0, 1, 3, 2], [0, 3, 2, 1], [0, 3, 1, 2]]

        References
        ==========

        .. [1] The Implementation of Various Algorithms for Permutation Groups in
               the Computer Algebra System: AXIOM, N.J. Doye, M.Sc. Thesis

        """
    idn = list(range(self.degree))
    order = 0
    element_list = [idn]
    set_element_list = {tuple(idn)}
    if af:
        yield idn
    else:
        yield _af_new(idn)
    gens = [p._array_form for p in self.generators]
    for i in range(len(gens)):
        D = element_list[:]
        N = [idn]
        while N:
            A = N
            N = []
            for a in A:
                for g in gens[:i + 1]:
                    ag = _af_rmul(a, g)
                    if tuple(ag) not in set_element_list:
                        for d in D:
                            order += 1
                            ap = _af_rmul(d, ag)
                            if af:
                                yield ap
                            else:
                                p = _af_new(ap)
                                yield p
                            element_list.append(ap)
                            set_element_list.add(tuple(ap))
                            N.append(ap)
    self._order = len(element_list)