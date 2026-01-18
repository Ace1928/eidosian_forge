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
def random_pr(self, gen_count=11, iterations=50, _random_prec=None):
    """Return a random group element using product replacement.

        Explanation
        ===========

        For the details of the product replacement algorithm, see
        ``_random_pr_init`` In ``random_pr`` the actual 'product replacement'
        is performed. Notice that if the attribute ``_random_gens``
        is empty, it needs to be initialized by ``_random_pr_init``.

        See Also
        ========

        _random_pr_init

        """
    if self._random_gens == []:
        self._random_pr_init(gen_count, iterations)
    random_gens = self._random_gens
    r = len(random_gens) - 1
    if _random_prec is None:
        s = randrange(r)
        t = randrange(r - 1)
        if t == s:
            t = r - 1
        x = choice([1, 2])
        e = choice([-1, 1])
    else:
        s = _random_prec['s']
        t = _random_prec['t']
        if t == s:
            t = r - 1
        x = _random_prec['x']
        e = _random_prec['e']
    if x == 1:
        random_gens[s] = _af_rmul(random_gens[s], _af_pow(random_gens[t], e))
        random_gens[r] = _af_rmul(random_gens[r], random_gens[s])
    else:
        random_gens[s] = _af_rmul(_af_pow(random_gens[t], e), random_gens[s])
        random_gens[r] = _af_rmul(random_gens[s], random_gens[r])
    return _af_new(random_gens[r])