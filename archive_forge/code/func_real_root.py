from sympy.core import Function, S, sympify, NumberKind
from sympy.utilities.iterables import sift
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.operations import LatticeOp, ShortCircuit
from sympy.core.function import (Application, Lambda,
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.power import Pow
from sympy.core.relational import Eq, Relational
from sympy.core.singleton import Singleton
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy
from sympy.core.rules import Transform
from sympy.core.logic import fuzzy_and, fuzzy_or, _torf
from sympy.core.traversal import walk
from sympy.core.numbers import Integer
from sympy.logic.boolalg import And, Or
def real_root(arg, n=None, evaluate=None):
    """Return the real *n*'th-root of *arg* if possible.

    Parameters
    ==========

    n : int or None, optional
        If *n* is ``None``, then all instances of
        $(-n)^{1/\\text{odd}}$ will be changed to $-n^{1/\\text{odd}}$.
        This will only create a real root of a principal root.
        The presence of other factors may cause the result to not be
        real.

    evaluate : bool, optional
        The parameter determines if the expression should be evaluated.
        If ``None``, its value is taken from
        ``global_parameters.evaluate``.

    Examples
    ========

    >>> from sympy import root, real_root

    >>> real_root(-8, 3)
    -2
    >>> root(-8, 3)
    2*(-1)**(1/3)
    >>> real_root(_)
    -2

    If one creates a non-principal root and applies real_root, the
    result will not be real (so use with caution):

    >>> root(-8, 3, 2)
    -2*(-1)**(2/3)
    >>> real_root(_)
    -2*(-1)**(2/3)

    See Also
    ========

    sympy.polys.rootoftools.rootof
    sympy.core.power.integer_nthroot
    root, sqrt
    """
    from sympy.functions.elementary.complexes import Abs, im, sign
    from sympy.functions.elementary.piecewise import Piecewise
    if n is not None:
        return Piecewise((root(arg, n, evaluate=evaluate), Or(Eq(n, S.One), Eq(n, S.NegativeOne))), (Mul(sign(arg), root(Abs(arg), n, evaluate=evaluate), evaluate=evaluate), And(Eq(im(arg), S.Zero), Eq(Mod(n, 2), S.One))), (root(arg, n, evaluate=evaluate), True))
    rv = sympify(arg)
    n1pow = Transform(lambda x: -(-x.base) ** x.exp, lambda x: x.is_Pow and x.base.is_negative and x.exp.is_Rational and (x.exp.p == 1) and x.exp.q % 2)
    return rv.xreplace(n1pow)