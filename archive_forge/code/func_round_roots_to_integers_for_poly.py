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
def round_roots_to_integers_for_poly(self, T):
    """
        For a given polynomial *T*, round the roots of this resolvent to the
        nearest integers.

        Explanation
        ===========

        None of the integers returned by this method is guaranteed to be a
        root of the resolvent; however, if the resolvent has any integer roots
        (for the given polynomial *T*), then they must be among these.

        If the coefficients of the resolvent are also desired, then this method
        should not be used. Instead, use the ``eval_for_poly`` method. This
        method may be significantly faster than ``eval_for_poly``.

        Parameters
        ==========

        T : :py:class:`~.Poly`

        Returns
        =======

        dict
            Keys are the indices of those permutations in ``self.s`` such that
            the corresponding root did round to a rational integer.

            Values are :ref:`ZZ`.


        """
    approx_roots_of_T = self.approximate_roots_of_poly(T, target='roots')
    approx_roots_of_self = [r(*approx_roots_of_T) for r in self.root_lambdas]
    return {i: self.round_mpf(r.real) for i, r in enumerate(approx_roots_of_self) if self.round_mpf(r.imag) == 0}