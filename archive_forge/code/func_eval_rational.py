from sympy.core.basic import Basic
from sympy.core import (S, Expr, Integer, Float, I, oo, Add, Lambda,
from sympy.core.cache import cacheit
from sympy.core.relational import is_le
from sympy.core.sorting import ordered
from sympy.polys.domains import QQ
from sympy.polys.polyerrors import (
from sympy.polys.polyfuncs import symmetrize, viete
from sympy.polys.polyroots import (
from sympy.polys.polytools import Poly, PurePoly, factor
from sympy.polys.rationaltools import together
from sympy.polys.rootisolation import (
from sympy.utilities import lambdify, public, sift, numbered_symbols
from mpmath import mpf, mpc, findroot, workprec
from mpmath.libmp.libmpf import dps_to_prec, prec_to_dps
from sympy.multipledispatch import dispatch
from itertools import chain
def eval_rational(self, dx=None, dy=None, n=15):
    """
        Return a Rational approximation of ``self`` that has real
        and imaginary component approximations that are within ``dx``
        and ``dy`` of the true values, respectively. Alternatively,
        ``n`` digits of precision can be specified.

        The interval is refined with bisection and is sure to
        converge. The root bounds are updated when the refinement
        is complete so recalculation at the same or lesser precision
        will not have to repeat the refinement and should be much
        faster.

        The following example first obtains Rational approximation to
        1e-8 accuracy for all roots of the 4-th order Legendre
        polynomial. Since the roots are all less than 1, this will
        ensure the decimal representation of the approximation will be
        correct (including rounding) to 6 digits:

        >>> from sympy import legendre_poly, Symbol
        >>> x = Symbol("x")
        >>> p = legendre_poly(4, x, polys=True)
        >>> r = p.real_roots()[-1]
        >>> r.eval_rational(10**-8).n(6)
        0.861136

        It is not necessary to a two-step calculation, however: the
        decimal representation can be computed directly:

        >>> r.evalf(17)
        0.86113631159405258

        """
    dy = dy or dx
    if dx:
        rtol = None
        dx = dx if isinstance(dx, Rational) else Rational(str(dx))
        dy = dy if isinstance(dy, Rational) else Rational(str(dy))
    else:
        rtol = S(10) ** (-(n + 2))
    interval = self._get_interval()
    while True:
        if self.is_real:
            if rtol:
                dx = abs(interval.center * rtol)
            interval = interval.refine_size(dx=dx)
            c = interval.center
            real = Rational(c)
            imag = S.Zero
            if not rtol or interval.dx < abs(c * rtol):
                break
        elif self.is_imaginary:
            if rtol:
                dy = abs(interval.center[1] * rtol)
                dx = 1
            interval = interval.refine_size(dx=dx, dy=dy)
            c = interval.center[1]
            imag = Rational(c)
            real = S.Zero
            if not rtol or interval.dy < abs(c * rtol):
                break
        else:
            if rtol:
                dx = abs(interval.center[0] * rtol)
                dy = abs(interval.center[1] * rtol)
            interval = interval.refine_size(dx, dy)
            c = interval.center
            real, imag = map(Rational, c)
            if not rtol or (interval.dx < abs(c[0] * rtol) and interval.dy < abs(c[1] * rtol)):
                break
    self._set_interval(interval)
    return real + I * imag