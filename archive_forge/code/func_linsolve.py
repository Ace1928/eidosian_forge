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
def linsolve(system, *symbols):
    """
    Solve system of $N$ linear equations with $M$ variables; both
    underdetermined and overdetermined systems are supported.
    The possible number of solutions is zero, one or infinite.
    Zero solutions throws a ValueError, whereas infinite
    solutions are represented parametrically in terms of the given
    symbols. For unique solution a :class:`~.FiniteSet` of ordered tuples
    is returned.

    All standard input formats are supported:
    For the given set of equations, the respective input types
    are given below:

    .. math:: 3x + 2y -   z = 1
    .. math:: 2x - 2y + 4z = -2
    .. math:: 2x -   y + 2z = 0

    * Augmented matrix form, ``system`` given below:

    $$ \\text{system} = \\left[{array}{cccc}
        3 &  2 & -1 &  1\\\\
        2 & -2 &  4 & -2\\\\
        2 & -1 &  2 &  0
        \\end{array}\\right] $$

    ::

        system = Matrix([[3, 2, -1, 1], [2, -2, 4, -2], [2, -1, 2, 0]])

    * List of equations form

    ::

        system  =  [3x + 2y - z - 1, 2x - 2y + 4z + 2, 2x - y + 2z]

    * Input $A$ and $b$ in matrix form (from $Ax = b$) are given as:

    $$ A = \\left[\\begin{array}{ccc}
        3 &  2 & -1 \\\\
        2 & -2 &  4 \\\\
        2 & -1 &  2
        \\end{array}\\right] \\ \\  b = \\left[\\begin{array}{c}
        1 \\\\ -2 \\\\ 0
        \\end{array}\\right] $$

    ::

        A = Matrix([[3, 2, -1], [2, -2, 4], [2, -1, 2]])
        b = Matrix([[1], [-2], [0]])
        system = (A, b)

    Symbols can always be passed but are actually only needed
    when 1) a system of equations is being passed and 2) the
    system is passed as an underdetermined matrix and one wants
    to control the name of the free variables in the result.
    An error is raised if no symbols are used for case 1, but if
    no symbols are provided for case 2, internally generated symbols
    will be provided. When providing symbols for case 2, there should
    be at least as many symbols are there are columns in matrix A.

    The algorithm used here is Gauss-Jordan elimination, which
    results, after elimination, in a row echelon form matrix.

    Returns
    =======

    A FiniteSet containing an ordered tuple of values for the
    unknowns for which the `system` has a solution. (Wrapping
    the tuple in FiniteSet is used to maintain a consistent
    output format throughout solveset.)

    Returns EmptySet, if the linear system is inconsistent.

    Raises
    ======

    ValueError
        The input is not valid.
        The symbols are not given.

    Examples
    ========

    >>> from sympy import Matrix, linsolve, symbols
    >>> x, y, z = symbols("x, y, z")
    >>> A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    >>> b = Matrix([3, 6, 9])
    >>> A
    Matrix([
    [1, 2,  3],
    [4, 5,  6],
    [7, 8, 10]])
    >>> b
    Matrix([
    [3],
    [6],
    [9]])
    >>> linsolve((A, b), [x, y, z])
    {(-1, 2, 0)}

    * Parametric Solution: In case the system is underdetermined, the
      function will return a parametric solution in terms of the given
      symbols. Those that are free will be returned unchanged. e.g. in
      the system below, `z` is returned as the solution for variable z;
      it can take on any value.

    >>> A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> b = Matrix([3, 6, 9])
    >>> linsolve((A, b), x, y, z)
    {(z - 1, 2 - 2*z, z)}

    If no symbols are given, internally generated symbols will be used.
    The ``tau0`` in the third position indicates (as before) that the third
    variable -- whatever it is named -- can take on any value:

    >>> linsolve((A, b))
    {(tau0 - 1, 2 - 2*tau0, tau0)}

    * List of equations as input

    >>> Eqns = [3*x + 2*y - z - 1, 2*x - 2*y + 4*z + 2, - x + y/2 - z]
    >>> linsolve(Eqns, x, y, z)
    {(1, -2, -2)}

    * Augmented matrix as input

    >>> aug = Matrix([[2, 1, 3, 1], [2, 6, 8, 3], [6, 8, 18, 5]])
    >>> aug
    Matrix([
    [2, 1,  3, 1],
    [2, 6,  8, 3],
    [6, 8, 18, 5]])
    >>> linsolve(aug, x, y, z)
    {(3/10, 2/5, 0)}

    * Solve for symbolic coefficients

    >>> a, b, c, d, e, f = symbols('a, b, c, d, e, f')
    >>> eqns = [a*x + b*y - c, d*x + e*y - f]
    >>> linsolve(eqns, x, y)
    {((-b*f + c*e)/(a*e - b*d), (a*f - c*d)/(a*e - b*d))}

    * A degenerate system returns solution as set of given
      symbols.

    >>> system = Matrix(([0, 0, 0], [0, 0, 0], [0, 0, 0]))
    >>> linsolve(system, x, y)
    {(x, y)}

    * For an empty system linsolve returns empty set

    >>> linsolve([], x)
    EmptySet

    * An error is raised if any nonlinearity is detected, even
      if it could be removed with expansion

    >>> linsolve([x*(1/x - 1)], x)
    Traceback (most recent call last):
    ...
    NonlinearError: nonlinear term: 1/x

    >>> linsolve([x*(y + 1)], x, y)
    Traceback (most recent call last):
    ...
    NonlinearError: nonlinear cross-term: x*(y + 1)

    >>> linsolve([x**2 - 1], x)
    Traceback (most recent call last):
    ...
    NonlinearError: nonlinear term: x**2
    """
    if not system:
        return S.EmptySet
    if symbols and hasattr(symbols[0], '__iter__'):
        symbols = symbols[0]
    sym_gen = isinstance(symbols, GeneratorType)
    dup_msg = 'duplicate symbols given'
    b = None
    if hasattr(system, '__iter__'):
        if len(system) == 2 and isinstance(system[0], MatrixBase):
            A, b = system
        if not isinstance(system[0], MatrixBase):
            if sym_gen or not symbols:
                raise ValueError(filldedent('\n                    When passing a system of equations, the explicit\n                    symbols for which a solution is being sought must\n                    be given as a sequence, too.\n                '))
            if len(set(symbols)) != len(symbols):
                raise ValueError(dup_msg)
            eqs = system
            eqs = [sympify(eq) for eq in eqs]
            try:
                sol = _linsolve(eqs, symbols)
            except PolyNonlinearError as exc:
                raise NonlinearError(str(exc))
            if sol is None:
                return S.EmptySet
            sol = FiniteSet(Tuple(*(sol.get(sym, sym) for sym in symbols)))
            return sol
    elif isinstance(system, MatrixBase) and (not (symbols and (not isinstance(symbols, GeneratorType)) and isinstance(symbols[0], MatrixBase))):
        A, b = (system[:, :-1], system[:, -1:])
    if b is None:
        raise ValueError('Invalid arguments')
    if sym_gen:
        symbols = [next(symbols) for i in range(A.cols)]
        symset = set(symbols)
        if any(symset & (A.free_symbols | b.free_symbols)):
            raise ValueError(filldedent("\n                At least one of the symbols provided\n                already appears in the system to be solved.\n                One way to avoid this is to use Dummy symbols in\n                the generator, e.g. numbered_symbols('%s', cls=Dummy)\n            " % symbols[0].name.rstrip('1234567890')))
        elif len(symset) != len(symbols):
            raise ValueError(dup_msg)
    if not symbols:
        symbols = [Dummy() for _ in range(A.cols)]
        name = _uniquely_named_symbol('tau', (A, b), compare=lambda i: str(i).rstrip('1234567890')).name
        gen = numbered_symbols(name)
    else:
        gen = None
    eqs = []
    rows = A.tolist()
    for rowi, bi in zip(rows, b):
        terms = [elem * sym for elem, sym in zip(rowi, symbols) if elem]
        terms.append(-bi)
        eqs.append(Add(*terms))
    eqs, ring = sympy_eqs_to_ring(eqs, symbols)
    sol = solve_lin_sys(eqs, ring, _raw=False)
    if sol is None:
        return S.EmptySet
    sol = FiniteSet(Tuple(*(sol.get(sym, sym) for sym in symbols)))
    if gen is not None:
        solsym = sol.free_symbols
        rep = {sym: next(gen) for sym in symbols if sym in solsym}
        sol = sol.subs(rep)
    return sol