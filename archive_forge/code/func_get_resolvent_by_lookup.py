from sympy.core.evalf import (
from sympy.core.symbol import symbols, Dummy
from sympy.polys.densetools import dup_eval
from sympy.polys.domains import ZZ
from sympy.polys.numberfields.resolvent_lookup import resolvent_coeff_lambdas
from sympy.polys.orderings import lex
from sympy.polys.polyroots import preprocess_roots
from sympy.polys.polytools import Poly
from sympy.polys.rings import xring
from sympy.polys.specialpolys import symmetric_poly
from sympy.utilities.lambdify import lambdify
from mpmath import MPContext
from mpmath.libmp.libmpf import prec_to_dps
def get_resolvent_by_lookup(T, number):
    """
    Use the lookup table, to return a resolvent (as dup) for a given
    polynomial *T*.

    Parameters
    ==========

    T : Poly
        The polynomial whose resolvent is needed

    number : int
        For some degrees, there are multiple resolvents.
        Use this to indicate which one you want.

    Returns
    =======

    dup

    """
    degree = T.degree()
    L = resolvent_coeff_lambdas[degree, number]
    T_coeffs = T.rep.rep[1:]
    return [ZZ(1)] + [c(*T_coeffs) for c in L]