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
def holzer(x, y, z, a, b, c):
    """
    Simplify the solution `(x, y, z)` of the equation
    `ax^2 + by^2 = cz^2` with `a, b, c > 0` and `z^2 \\geq \\mid ab \\mid` to
    a new reduced solution `(x', y', z')` such that `z'^2 \\leq \\mid ab \\mid`.

    The algorithm is an interpretation of Mordell's reduction as described
    on page 8 of Cremona and Rusin's paper [1]_ and the work of Mordell in
    reference [2]_.

    References
    ==========

    .. [1] Efficient Solution of Rational Conices, J. E. Cremona and D. Rusin,
           Mathematics of Computation, Volume 00, Number 0.
    .. [2] Diophantine Equations, L. J. Mordell, page 48.

    """
    if _odd(c):
        k = 2 * c
    else:
        k = c // 2
    small = a * b * c
    step = 0
    while True:
        t1, t2, t3 = (a * x ** 2, b * y ** 2, c * z ** 2)
        if t1 + t2 != t3:
            if step == 0:
                raise ValueError('bad starting solution')
            break
        x_0, y_0, z_0 = (x, y, z)
        if max(t1, t2, t3) <= small:
            break
        uv = u, v = base_solution_linear(k, y_0, -x_0)
        if None in uv:
            break
        p, q = (-(a * u * x_0 + b * v * y_0), c * z_0)
        r = Rational(p, q)
        if _even(c):
            w = _nint_or_floor(p, q)
            assert abs(w - r) <= S.Half
        else:
            w = p // q
            if _odd(a * u + b * v + c * w):
                w += 1
            assert abs(w - r) <= S.One
        A = a * u ** 2 + b * v ** 2 + c * w ** 2
        B = a * u * x_0 + b * v * y_0 + c * w * z_0
        x = Rational(x_0 * A - 2 * u * B, k)
        y = Rational(y_0 * A - 2 * v * B, k)
        z = Rational(z_0 * A - 2 * w * B, k)
        assert all((i.is_Integer for i in (x, y, z)))
        step += 1
    return tuple([int(i) for i in (x_0, y_0, z_0)])