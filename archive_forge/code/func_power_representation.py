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
def power_representation(n, p, k, zeros=False):
    """
    Returns a generator for finding k-tuples of integers,
    `(n_{1}, n_{2}, . . . n_{k})`, such that
    `n = n_{1}^p + n_{2}^p + . . . n_{k}^p`.

    Usage
    =====

    ``power_representation(n, p, k, zeros)``: Represent non-negative number
    ``n`` as a sum of ``k`` ``p``\\ th powers. If ``zeros`` is true, then the
    solutions is allowed to contain zeros.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import power_representation

    Represent 1729 as a sum of two cubes:

    >>> f = power_representation(1729, 3, 2)
    >>> next(f)
    (9, 10)
    >>> next(f)
    (1, 12)

    If the flag `zeros` is True, the solution may contain tuples with
    zeros; any such solutions will be generated after the solutions
    without zeros:

    >>> list(power_representation(125, 2, 3, zeros=True))
    [(5, 6, 8), (3, 4, 10), (0, 5, 10), (0, 2, 11)]

    For even `p` the `permute_sign` function can be used to get all
    signed values:

    >>> from sympy.utilities.iterables import permute_signs
    >>> list(permute_signs((1, 12)))
    [(1, 12), (-1, 12), (1, -12), (-1, -12)]

    All possible signed permutations can also be obtained:

    >>> from sympy.utilities.iterables import signed_permutations
    >>> list(signed_permutations((1, 12)))
    [(1, 12), (-1, 12), (1, -12), (-1, -12), (12, 1), (-12, 1), (12, -1), (-12, -1)]
    """
    n, p, k = [as_int(i) for i in (n, p, k)]
    if n < 0:
        if p % 2:
            for t in power_representation(-n, p, k, zeros):
                yield tuple((-i for i in t))
        return
    if p < 1 or k < 1:
        raise ValueError(filldedent('\n    Expecting positive integers for `(p, k)`, but got `(%s, %s)`' % (p, k)))
    if n == 0:
        if zeros:
            yield ((0,) * k)
        return
    if k == 1:
        if p == 1:
            yield (n,)
        else:
            be = perfect_power(n)
            if be:
                b, e = be
                d, r = divmod(e, p)
                if not r:
                    yield (b ** d,)
        return
    if p == 1:
        for t in partition(n, k, zeros=zeros):
            yield t
        return
    if p == 2:
        feasible = _can_do_sum_of_squares(n, k)
        if not feasible:
            return
        if not zeros and n > 33 and (k >= 5) and (k <= n) and (n - k in (13, 10, 7, 5, 4, 2, 1)):
            'Todd G. Will, "When Is n^2 a Sum of k Squares?", [online].\n                Available: https://www.maa.org/sites/default/files/Will-MMz-201037918.pdf'
            return
        if feasible is not True:
            yield prime_as_sum_of_two_squares(n)
            return
    if k == 2 and p > 2:
        be = perfect_power(n)
        if be and be[1] % p == 0:
            return
    if n >= k:
        a = integer_nthroot(n - (k - 1), p)[0]
        for t in pow_rep_recursive(a, k, n, [], p):
            yield tuple(reversed(t))
    if zeros:
        a = integer_nthroot(n, p)[0]
        for i in range(1, k):
            for t in pow_rep_recursive(a, i, n, [], p):
                yield tuple(reversed(t + (0,) * (k - i)))