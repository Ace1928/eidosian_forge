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
def named_poly(n, f, K, name, x, polys):
    """Common interface to the low-level polynomial generating functions
    in orthopolys and appellseqs.

    Parameters
    ==========

    n : int
        Index of the polynomial, which may or may not equal its degree.
    f : callable
        Low-level generating function to use.
    K : Domain or None
        Domain in which to perform the computations. If None, use the smallest
        field containing the rationals and the extra parameters of x (see below).
    name : str
        Name of an arbitrary individual polynomial in the sequence generated
        by f, only used in the error message for invalid n.
    x : seq
        The first element of this argument is the main variable of all
        polynomials in this sequence. Any further elements are extra
        parameters required by f.
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.
    """
    if n < 0:
        raise ValueError('Cannot generate %s of index %s' % (name, n))
    head, tail = (x[0], x[1:])
    if K is None:
        K, tail = construct_domain(tail, field=True)
    poly = DMP(f(int(n), *tail, K), K)
    if head is None:
        poly = PurePoly.new(poly, Dummy('x'))
    else:
        poly = Poly.new(poly, head)
    return poly if polys else poly.as_expr()