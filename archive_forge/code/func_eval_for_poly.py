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
def eval_for_poly(self, T, find_integer_root=False):
    """
        Compute the integer values of the coefficients of this resolvent, when
        plugging in the roots of a given polynomial.

        Parameters
        ==========

        T : :py:class:`~.Poly`

        find_integer_root : ``bool``, default ``False``
            If ``True``, then also determine whether the resolvent has an
            integer root, and return the first one found, along with its
            index, i.e. the index of the permutation ``self.s[i]`` it
            corresponds to.

        Returns
        =======

        Tuple ``(R, a, i)``

            ``R`` is this resolvent as a dense univariate polynomial over
            :ref:`ZZ`, i.e. a list of :ref:`ZZ`.

            If *find_integer_root* was ``True``, then ``a`` and ``i`` are the
            first integer root found, and its index, if one exists.
            Otherwise ``a`` and ``i`` are both ``None``.

        """
    approx_roots_of_T = self.approximate_roots_of_poly(T, target='coeffs')
    approx_roots_of_self = [r(*approx_roots_of_T) for r in self.root_lambdas]
    approx_coeffs_of_self = [c(*approx_roots_of_self) for c in self.esf_lambdas]
    R = []
    for c in approx_coeffs_of_self:
        if self.round_mpf(c.imag) != 0:
            raise ResolventException(f'Got non-integer coeff for resolvent: {c}')
        R.append(self.round_mpf(c.real))
    a0, i0 = (None, None)
    if find_integer_root:
        for i, r in enumerate(approx_roots_of_self):
            if self.round_mpf(r.imag) != 0:
                continue
            if not dup_eval(R, (a := self.round_mpf(r.real)), ZZ):
                a0, i0 = (a, i)
                break
    return (R, a0, i0)