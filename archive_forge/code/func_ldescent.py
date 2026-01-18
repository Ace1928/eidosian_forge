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
def ldescent(A, B):
    """
    Return a non-trivial solution to `w^2 = Ax^2 + By^2` using
    Lagrange's method; return None if there is no such solution.
    .

    Here, `A \\neq 0` and `B \\neq 0` and `A` and `B` are square free. Output a
    tuple `(w_0, x_0, y_0)` which is a solution to the above equation.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import ldescent
    >>> ldescent(1, 1) # w^2 = x^2 + y^2
    (1, 1, 0)
    >>> ldescent(4, -7) # w^2 = 4x^2 - 7y^2
    (2, -1, 0)

    This means that `x = -1, y = 0` and `w = 2` is a solution to the equation
    `w^2 = 4x^2 - 7y^2`

    >>> ldescent(5, -1) # w^2 = 5x^2 - y^2
    (2, 1, -1)

    References
    ==========

    .. [1] The algorithmic resolution of Diophantine equations, Nigel P. Smart,
           London Mathematical Society Student Texts 41, Cambridge University
           Press, Cambridge, 1998.
    .. [2] Efficient Solution of Rational Conices, J. E. Cremona and D. Rusin,
           [online], Available:
           https://nottingham-repository.worktribe.com/output/1023265/efficient-solution-of-rational-conics
    """
    if abs(A) > abs(B):
        w, y, x = ldescent(B, A)
        return (w, x, y)
    if A == 1:
        return (1, 1, 0)
    if B == 1:
        return (1, 0, 1)
    if B == -1:
        return
    r = sqrt_mod(A, B)
    Q = (r ** 2 - A) // B
    if Q == 0:
        B_0 = 1
        d = 0
    else:
        div = divisors(Q)
        B_0 = None
        for i in div:
            sQ, _exact = integer_nthroot(abs(Q) // i, 2)
            if _exact:
                B_0, d = (sign(Q) * i, sQ)
                break
    if B_0 is not None:
        W, X, Y = ldescent(A, B_0)
        return _remove_gcd(-A * X + r * W, r * X - W, Y * (B_0 * d))