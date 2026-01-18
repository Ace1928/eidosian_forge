from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.functions import DiracDelta, Heaviside
from .integrals import Integral, integrate
def deltaintegrate(f, x):
    """
    deltaintegrate(f, x)

    Explanation
    ===========

    The idea for integration is the following:

    - If we are dealing with a DiracDelta expression, i.e. DiracDelta(g(x)),
      we try to simplify it.

      If we could simplify it, then we integrate the resulting expression.
      We already know we can integrate a simplified expression, because only
      simple DiracDelta expressions are involved.

      If we couldn't simplify it, there are two cases:

      1) The expression is a simple expression: we return the integral,
         taking care if we are dealing with a Derivative or with a proper
         DiracDelta.

      2) The expression is not simple (i.e. DiracDelta(cos(x))): we can do
         nothing at all.

    - If the node is a multiplication node having a DiracDelta term:

      First we expand it.

      If the expansion did work, then we try to integrate the expansion.

      If not, we try to extract a simple DiracDelta term, then we have two
      cases:

      1) We have a simple DiracDelta term, so we return the integral.

      2) We didn't have a simple term, but we do have an expression with
         simplified DiracDelta terms, so we integrate this expression.

    Examples
    ========

        >>> from sympy.abc import x, y, z
        >>> from sympy.integrals.deltafunctions import deltaintegrate
        >>> from sympy import sin, cos, DiracDelta
        >>> deltaintegrate(x*sin(x)*cos(x)*DiracDelta(x - 1), x)
        sin(1)*cos(1)*Heaviside(x - 1)
        >>> deltaintegrate(y**2*DiracDelta(x - z)*DiracDelta(y - z), y)
        z**2*DiracDelta(x - z)*Heaviside(y - z)

    See Also
    ========

    sympy.functions.special.delta_functions.DiracDelta
    sympy.integrals.integrals.Integral
    """
    if not f.has(DiracDelta):
        return None
    if f.func == DiracDelta:
        h = f.expand(diracdelta=True, wrt=x)
        if h == f:
            if f.is_simple(x):
                if len(f.args) <= 1 or f.args[1] == 0:
                    return Heaviside(f.args[0])
                else:
                    return DiracDelta(f.args[0], f.args[1] - 1) / f.args[0].as_poly().LC()
        else:
            fh = integrate(h, x)
            return fh
    elif f.is_Mul or f.is_Pow:
        g = f.expand()
        if f != g:
            fh = integrate(g, x)
            if fh is not None and (not isinstance(fh, Integral)):
                return fh
        else:
            deltaterm, rest_mult = change_mul(f, x)
            if not deltaterm:
                if rest_mult:
                    fh = integrate(rest_mult, x)
                    return fh
            else:
                from sympy.solvers import solve
                deltaterm = deltaterm.expand(diracdelta=True, wrt=x)
                if deltaterm.is_Mul:
                    deltaterm, rest_mult_2 = change_mul(deltaterm, x)
                    rest_mult = rest_mult * rest_mult_2
                point = solve(deltaterm.args[0], x)[0]
                n = 0 if len(deltaterm.args) == 1 else deltaterm.args[1]
                m = 0
                while n >= 0:
                    r = S.NegativeOne ** n * rest_mult.diff(x, n).subs(x, point)
                    if r.is_zero:
                        n -= 1
                        m += 1
                    elif m == 0:
                        return r * Heaviside(x - point)
                    else:
                        return r * DiracDelta(x, m - 1)
                return S.Zero
    return None