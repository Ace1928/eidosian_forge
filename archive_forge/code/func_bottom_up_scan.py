from functools import reduce
from sympy.core.add import Add
from sympy.core.exprtools import Factors
from sympy.core.function import expand_mul, expand_multinomial, _mexpand
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, pi, _illegal)
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.core.traversal import preorder_traversal
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt, cbrt
from sympy.functions.elementary.trigonometric import cos, sin, tan
from sympy.ntheory.factor_ import divisors
from sympy.utilities.iterables import subsets
from sympy.polys.domains import ZZ, QQ, FractionField
from sympy.polys.orthopolys import dup_chebyshevt
from sympy.polys.polyerrors import (
from sympy.polys.polytools import (
from sympy.polys.polyutils import dict_from_expr, expr_from_dict
from sympy.polys.ring_series import rs_compose_add
from sympy.polys.rings import ring
from sympy.polys.rootoftools import CRootOf
from sympy.polys.specialpolys import cyclotomic_poly
from sympy.utilities import (
def bottom_up_scan(ex):
    """
        Transform a given algebraic expression *ex* into a multivariate
        polynomial, by introducing fresh variables with defining equations.

        Explanation
        ===========

        The critical elements of the algebraic expression *ex* are root
        extractions, instances of :py:class:`~.AlgebraicNumber`, and negative
        powers.

        When we encounter a root extraction or an :py:class:`~.AlgebraicNumber`
        we replace this expression with a fresh variable ``a_i``, and record
        the defining polynomial for ``a_i``. For example, if ``a_0**(1/3)``
        occurs, we will replace it with ``a_1``, and record the new defining
        polynomial ``a_1**3 - a_0``.

        When we encounter a negative power we transform it into a positive
        power by algebraically inverting the base. This means computing the
        minimal polynomial in ``x`` for the base, inverting ``x`` modulo this
        poly (which generates a new polynomial) and then substituting the
        original base expression for ``x`` in this last polynomial.

        We return the transformed expression, and we record the defining
        equations for new symbols using the ``update_mapping()`` function.

        """
    if ex.is_Atom:
        if ex is S.ImaginaryUnit:
            if ex not in mapping:
                return update_mapping(ex, 2, 1)
            else:
                return symbols[ex]
        elif ex.is_Rational:
            return ex
    elif ex.is_Add:
        return Add(*[bottom_up_scan(g) for g in ex.args])
    elif ex.is_Mul:
        return Mul(*[bottom_up_scan(g) for g in ex.args])
    elif ex.is_Pow:
        if ex.exp.is_Rational:
            if ex.exp < 0:
                minpoly_base = _minpoly_groebner(ex.base, x, cls)
                inverse = invert(x, minpoly_base).as_expr()
                base_inv = inverse.subs(x, ex.base).expand()
                if ex.exp == -1:
                    return bottom_up_scan(base_inv)
                else:
                    ex = base_inv ** (-ex.exp)
            if not ex.exp.is_Integer:
                base, exp = ((ex.base ** ex.exp.p).expand(), Rational(1, ex.exp.q))
            else:
                base, exp = (ex.base, ex.exp)
            base = bottom_up_scan(base)
            expr = base ** exp
            if expr not in mapping:
                if exp.is_Integer:
                    return expr.expand()
                else:
                    return update_mapping(expr, 1 / exp, -base)
            else:
                return symbols[expr]
    elif ex.is_AlgebraicNumber:
        if ex not in mapping:
            return update_mapping(ex, ex.minpoly_of_element())
        else:
            return symbols[ex]
    raise NotAlgebraic('%s does not seem to be an algebraic number' % ex)