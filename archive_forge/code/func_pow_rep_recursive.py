from sympy.core.add import Add
from sympy.core.assumptions import check_assumptions
from sympy.core.containers import Tuple
from sympy.core.exprtools import factor_terms
from sympy.core.function import _mexpand
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.numbers import igcdex, ilcm, igcd
from sympy.core.power import integer_nthroot, isqrt
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Symbol, symbols
from sympy.core.sympify import _sympify
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.ntheory.factor_ import (
from sympy.ntheory.generate import nextprime
from sympy.ntheory.primetest import is_square, isprime
from sympy.ntheory.residue_ntheory import sqrt_mod
from sympy.polys.polyerrors import GeneratorsNeeded
from sympy.polys.polytools import Poly, factor_list
from sympy.simplify.simplify import signsimp
from sympy.solvers.solveset import solveset_real
from sympy.utilities import numbered_symbols
from sympy.utilities.misc import as_int, filldedent
from sympy.utilities.iterables import (is_sequence, subsets, permute_signs,
def pow_rep_recursive(n_i, k, n_remaining, terms, p):
    if n_i <= 0 or k <= 0:
        return
    if n_remaining < k:
        return
    if k * pow(n_i, p) < n_remaining:
        return
    if k == 0 and n_remaining == 0:
        yield tuple(terms)
    elif k == 1:
        next_term, exact = integer_nthroot(n_remaining, p)
        if exact and next_term <= n_i:
            yield tuple(terms + [next_term])
        return
    elif n_i >= 1 and k > 0:
        for next_term in range(1, n_i + 1):
            residual = n_remaining - pow(next_term, p)
            if residual < 0:
                break
            yield from pow_rep_recursive(next_term, k - 1, residual, terms + [next_term], p)