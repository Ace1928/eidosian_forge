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
def eval_approx(self, n, return_mpmath=False):
    """Evaluate this complex root to the given precision.

        This uses secant method and root bounds are used to both
        generate an initial guess and to check that the root
        returned is valid. If ever the method converges outside the
        root bounds, the bounds will be made smaller and updated.
        """
    prec = dps_to_prec(n)
    with workprec(prec):
        g = self.poly.gen
        if not g.is_Symbol:
            d = Dummy('x')
            if self.is_imaginary:
                d *= I
            func = lambdify(d, self.expr.subs(g, d))
        else:
            expr = self.expr
            if self.is_imaginary:
                expr = self.expr.subs(g, I * g)
            func = lambdify(g, expr)
        interval = self._get_interval()
        while True:
            if self.is_real:
                a = mpf(str(interval.a))
                b = mpf(str(interval.b))
                if a == b:
                    root = a
                    break
                x0 = mpf(str(interval.center))
                x1 = x0 + mpf(str(interval.dx)) / 4
            elif self.is_imaginary:
                a = mpf(str(interval.ay))
                b = mpf(str(interval.by))
                if a == b:
                    root = mpc(mpf('0'), a)
                    break
                x0 = mpf(str(interval.center[1]))
                x1 = x0 + mpf(str(interval.dy)) / 4
            else:
                ax = mpf(str(interval.ax))
                bx = mpf(str(interval.bx))
                ay = mpf(str(interval.ay))
                by = mpf(str(interval.by))
                if ax == bx and ay == by:
                    root = mpc(ax, ay)
                    break
                x0 = mpc(*map(str, interval.center))
                x1 = x0 + mpc(*map(str, (interval.dx, interval.dy))) / 4
            try:
                root = findroot(func, (x0, x1))
                if self.is_real or self.is_imaginary:
                    if not bool(root.imag) == self.is_real and a <= root <= b:
                        if self.is_imaginary:
                            root = mpc(mpf('0'), root.real)
                        break
                elif ax <= root.real <= bx and ay <= root.imag <= by:
                    break
            except (UnboundLocalError, ValueError):
                pass
            interval = interval.refine()
    self._set_interval(interval)
    if return_mpmath:
        return root
    return Float._new(root.real._mpf_, prec) + I * Float._new(root.imag._mpf_, prec)