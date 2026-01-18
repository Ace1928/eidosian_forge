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
def parametrize_ternary_quadratic(eq):
    """
    Returns the parametrized general solution for the ternary quadratic
    equation ``eq`` which has the form
    `ax^2 + by^2 + cz^2 + fxy + gyz + hxz = 0`.

    Examples
    ========

    >>> from sympy import Tuple, ordered
    >>> from sympy.abc import x, y, z
    >>> from sympy.solvers.diophantine.diophantine import parametrize_ternary_quadratic

    The parametrized solution may be returned with three parameters:

    >>> parametrize_ternary_quadratic(2*x**2 + y**2 - 2*z**2)
    (p**2 - 2*q**2, -2*p**2 + 4*p*q - 4*p*r - 4*q**2, p**2 - 4*p*q + 2*q**2 - 4*q*r)

    There might also be only two parameters:

    >>> parametrize_ternary_quadratic(4*x**2 + 2*y**2 - 3*z**2)
    (2*p**2 - 3*q**2, -4*p**2 + 12*p*q - 6*q**2, 4*p**2 - 8*p*q + 6*q**2)

    Notes
    =====

    Consider ``p`` and ``q`` in the previous 2-parameter
    solution and observe that more than one solution can be represented
    by a given pair of parameters. If `p` and ``q`` are not coprime, this is
    trivially true since the common factor will also be a common factor of the
    solution values. But it may also be true even when ``p`` and
    ``q`` are coprime:

    >>> sol = Tuple(*_)
    >>> p, q = ordered(sol.free_symbols)
    >>> sol.subs([(p, 3), (q, 2)])
    (6, 12, 12)
    >>> sol.subs([(q, 1), (p, 1)])
    (-1, 2, 2)
    >>> sol.subs([(q, 0), (p, 1)])
    (2, -4, 4)
    >>> sol.subs([(q, 1), (p, 0)])
    (-3, -6, 6)

    Except for sign and a common factor, these are equivalent to
    the solution of (1, 2, 2).

    References
    ==========

    .. [1] The algorithmic resolution of Diophantine equations, Nigel P. Smart,
           London Mathematical Society Student Texts 41, Cambridge University
           Press, Cambridge, 1998.

    """
    var, coeff, diop_type = classify_diop(eq, _dict=False)
    if diop_type in ('homogeneous_ternary_quadratic', 'homogeneous_ternary_quadratic_normal'):
        x_0, y_0, z_0 = list(_diop_ternary_quadratic(var, coeff))[0]
        return _parametrize_ternary_quadratic((x_0, y_0, z_0), var, coeff)