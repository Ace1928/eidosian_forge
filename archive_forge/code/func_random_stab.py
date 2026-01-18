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
def random_stab(self, alpha, schreier_vector=None, _random_prec=None):
    """Random element from the stabilizer of ``alpha``.

        The schreier vector for ``alpha`` is an optional argument used
        for speeding up repeated calls. The algorithm is described in [1], p.81

        See Also
        ========

        random_pr, orbit_rep

        """
    if schreier_vector is None:
        schreier_vector = self.schreier_vector(alpha)
    if _random_prec is None:
        rand = self.random_pr()
    else:
        rand = _random_prec['rand']
    beta = rand(alpha)
    h = self.orbit_rep(alpha, beta, schreier_vector)
    return rmul(~h, rand)