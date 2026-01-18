from functools import wraps, reduce
from operator import mul
from typing import Optional
from sympy.core import (
from sympy.core.basic import Basic
from sympy.core.decorators import _sympifyit
from sympy.core.exprtools import Factors, factor_nc, factor_terms
from sympy.core.evalf import (
from sympy.core.function import Derivative
from sympy.core.mul import Mul, _keep_coeff
from sympy.core.numbers import ilcm, I, Integer, equal_valued
from sympy.core.relational import Relational, Equality
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify, _sympify
from sympy.core.traversal import preorder_traversal, bottom_up
from sympy.logic.boolalg import BooleanAtom
from sympy.polys import polyoptions as options
from sympy.polys.constructor import construct_domain
from sympy.polys.domains import FF, QQ, ZZ
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.fglmtools import matrix_fglm
from sympy.polys.groebnertools import groebner as _groebner
from sympy.polys.monomials import Monomial
from sympy.polys.orderings import monomial_key
from sympy.polys.polyclasses import DMP, DMF, ANP
from sympy.polys.polyerrors import (
from sympy.polys.polyutils import (
from sympy.polys.rationaltools import together
from sympy.polys.rootisolation import dup_isolate_real_roots_list
from sympy.utilities import group, public, filldedent
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import iterable, sift
import sympy.polys
import mpmath
from mpmath.libmp.libhyper import NoConvergence
def nth_power_roots_poly(f, n):
    """
        Construct a polynomial with n-th powers of roots of ``f``.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> f = Poly(x**4 - x**2 + 1)

        >>> f.nth_power_roots_poly(2)
        Poly(x**4 - 2*x**3 + 3*x**2 - 2*x + 1, x, domain='ZZ')
        >>> f.nth_power_roots_poly(3)
        Poly(x**4 + 2*x**2 + 1, x, domain='ZZ')
        >>> f.nth_power_roots_poly(4)
        Poly(x**4 + 2*x**3 + 3*x**2 + 2*x + 1, x, domain='ZZ')
        >>> f.nth_power_roots_poly(12)
        Poly(x**4 - 4*x**3 + 6*x**2 - 4*x + 1, x, domain='ZZ')

        """
    if f.is_multivariate:
        raise MultivariatePolynomialError('must be a univariate polynomial')
    N = sympify(n)
    if N.is_Integer and N >= 1:
        n = int(N)
    else:
        raise ValueError("'n' must an integer and n >= 1, got %s" % n)
    x = f.gen
    t = Dummy('t')
    r = f.resultant(f.__class__.from_expr(x ** n - t, x, t))
    return r.replace(t, x)