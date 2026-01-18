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
def get_prec(self, M, target='coeffs'):
    """
        For a given upper bound *M* on the magnitude of the complex numbers to
        be plugged in for this resolvent's symbols, compute a sufficient
        precision for evaluating those complex numbers, such that the
        coefficients, or the integer roots, of the resolvent can be determined.

        Parameters
        ==========

        M : real number
            Upper bound on magnitude of the complex numbers to be plugged in.

        target : str, 'coeffs' or 'roots', default='coeffs'
            Name the task for which a sufficient precision is desired.
            This is either determining the coefficients of the resolvent
            ('coeffs') or determining its possible integer roots ('roots').
            The latter may require significantly lower precision.

        Returns
        =======

        int $m$
            such that $2^{-m}$ is a sufficient upper bound on the
            error in approximating the complex numbers to be plugged in.

        """
    M = max(M, 2)
    f = self.coeff_prec_func if target == 'coeffs' else self.root_prec_func
    r, _, _, _ = evalf(2 * f(M), 1, {})
    return fastlog(r) + 1