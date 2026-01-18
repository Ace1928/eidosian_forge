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
def same_root(f, a, b):
    """
        Decide whether two roots of this polynomial are equal.

        Examples
        ========

        >>> from sympy import Poly, cyclotomic_poly, exp, I, pi
        >>> f = Poly(cyclotomic_poly(5))
        >>> r0 = exp(2*I*pi/5)
        >>> indices = [i for i, r in enumerate(f.all_roots()) if f.same_root(r, r0)]
        >>> print(indices)
        [3]

        Raises
        ======

        DomainError
            If the domain of the polynomial is not :ref:`ZZ`, :ref:`QQ`,
            :ref:`RR`, or :ref:`CC`.
        MultivariatePolynomialError
            If the polynomial is not univariate.
        PolynomialError
            If the polynomial is of degree < 2.

        """
    if f.is_multivariate:
        raise MultivariatePolynomialError('Must be a univariate polynomial')
    dom_delta_sq = f.rep.mignotte_sep_bound_squared()
    delta_sq = f.domain.get_field().to_sympy(dom_delta_sq)
    eps_sq = delta_sq / 9
    r, _, _, _ = evalf(1 / eps_sq, 1, {})
    n = fastlog(r)
    m = n // 2 + n % 2
    ev = lambda x: quad_to_mpmath(_evalf_with_bounded_error(x, m=m))
    A, B = (ev(a), ev(b))
    return (A.real - B.real) ** 2 + (A.imag - B.imag) ** 2 < eps_sq