from sympy.core.sympify import sympify
from sympy.core import (S, Pow, Dummy, pi, Expr, Wild, Mul, Equality,
from sympy.core.containers import Tuple
from sympy.core.function import (Lambda, expand_complex, AppliedUndef,
from sympy.core.mod import Mod
from sympy.core.numbers import igcd, I, Number, Rational, oo, ilcm
from sympy.core.power import integer_log
from sympy.core.relational import Eq, Ne, Relational
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Symbol, _uniquely_named_symbol
from sympy.core.sympify import _sympify
from sympy.polys.matrices.linsolve import _linear_eq_to_dict
from sympy.polys.polyroots import UnsolvableFactorError
from sympy.simplify.simplify import simplify, fraction, trigsimp, nsimplify
from sympy.simplify import powdenest, logcombine
from sympy.functions import (log, tan, cot, sin, cos, sec, csc, exp,
from sympy.functions.elementary.complexes import Abs, arg, re, im
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.miscellaneous import real_root
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.logic.boolalg import And, BooleanTrue
from sympy.sets import (FiniteSet, imageset, Interval, Intersection,
from sympy.sets.sets import Set, ProductSet
from sympy.matrices import zeros, Matrix, MatrixBase
from sympy.ntheory import totient
from sympy.ntheory.factor_ import divisors
from sympy.ntheory.residue_ntheory import discrete_log, nthroot_mod
from sympy.polys import (roots, Poly, degree, together, PolynomialError,
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polytools import invert, groebner, poly
from sympy.polys.solvers import (sympy_eqs_to_ring, solve_lin_sys,
from sympy.polys.matrices.linsolve import _linsolve
from sympy.solvers.solvers import (checksol, denoms, unrad,
from sympy.solvers.polysys import solve_poly_system
from sympy.utilities import filldedent
from sympy.utilities.iterables import (numbered_symbols, has_dups,
from sympy.calculus.util import periodicity, continuous_domain, function_range
from types import GeneratorType
def linear_eq_to_matrix(equations, *symbols):
    """
    Converts a given System of Equations into Matrix form.
    Here `equations` must be a linear system of equations in
    `symbols`. Element ``M[i, j]`` corresponds to the coefficient
    of the jth symbol in the ith equation.

    The Matrix form corresponds to the augmented matrix form.
    For example:

    .. math:: 4x + 2y + 3z  = 1
    .. math:: 3x +  y +  z  = -6
    .. math:: 2x + 4y + 9z  = 2

    This system will return $A$ and $b$ as:

    $$ A = \\left[\\begin{array}{ccc}
        4 & 2 & 3 \\\\
        3 & 1 & 1 \\\\
        2 & 4 & 9
        \\end{array}\\right] \\ \\  b = \\left[\\begin{array}{c}
        1 \\\\ -6 \\\\ 2
        \\end{array}\\right] $$

    The only simplification performed is to convert
    ``Eq(a, b)`` $\\Rightarrow a - b$.

    Raises
    ======

    NonlinearError
        The equations contain a nonlinear term.
    ValueError
        The symbols are not given or are not unique.

    Examples
    ========

    >>> from sympy import linear_eq_to_matrix, symbols
    >>> c, x, y, z = symbols('c, x, y, z')

    The coefficients (numerical or symbolic) of the symbols will
    be returned as matrices:

        >>> eqns = [c*x + z - 1 - c, y + z, x - y]
        >>> A, b = linear_eq_to_matrix(eqns, [x, y, z])
        >>> A
        Matrix([
        [c,  0, 1],
        [0,  1, 1],
        [1, -1, 0]])
        >>> b
        Matrix([
        [c + 1],
        [    0],
        [    0]])

    This routine does not simplify expressions and will raise an error
    if nonlinearity is encountered:

            >>> eqns = [
            ...     (x**2 - 3*x)/(x - 3) - 3,
            ...     y**2 - 3*y - y*(y - 4) + x - 4]
            >>> linear_eq_to_matrix(eqns, [x, y])
            Traceback (most recent call last):
            ...
            NonlinearError:
            symbol-dependent term can be ignored using `strict=False`

        Simplifying these equations will discard the removable singularity
        in the first and reveal the linear structure of the second:

            >>> [e.simplify() for e in eqns]
            [x - 3, x + y - 4]

        Any such simplification needed to eliminate nonlinear terms must
        be done *before* calling this routine.
    """
    if not symbols:
        raise ValueError(filldedent('\n            Symbols must be given, for which coefficients\n            are to be found.\n            '))
    if hasattr(symbols[0], '__iter__'):
        symbols = symbols[0]
    if has_dups(symbols):
        raise ValueError('Symbols must be unique')
    equations = sympify(equations)
    if isinstance(equations, MatrixBase):
        equations = list(equations)
    elif isinstance(equations, (Expr, Eq)):
        equations = [equations]
    elif not is_sequence(equations):
        raise ValueError(filldedent('\n            Equation(s) must be given as a sequence, Expr,\n            Eq or Matrix.\n            '))
    try:
        eq, c = _linear_eq_to_dict(equations, symbols)
    except PolyNonlinearError as err:
        raise NonlinearError(str(err))
    n, m = shape = (len(eq), len(symbols))
    ix = dict(zip(symbols, range(m)))
    A = zeros(*shape)
    for row, d in enumerate(eq):
        for k in d:
            col = ix[k]
            A[row, col] = d[k]
    b = Matrix(n, 1, [-i for i in c])
    return (A, b)