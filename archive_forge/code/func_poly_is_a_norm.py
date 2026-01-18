from ..sage_helper import _within_sage, sage_method
from .. import SnapPy
def poly_is_a_norm(g):
    """
    Return whether the polynomial g(t) over a CyclotomicField is equal to
    (const) f(t) fbar(t) where fbar is poly_involution(f)::

       sage: K = CyclotomicField(5, 'z')
       sage: R = PolynomialRing(K, 't')
       sage: z, t = K.gen(), R.gen()
       sage: f = z*t**2 + (1/z)*t + 1
       sage: fbar = poly_involution(f)
       sage: poly_is_a_norm(z**2 * f * fbar * (t - 1)**2)
       True
       sage: poly_is_a_norm(f**2 * fbar)
       False
       sage: poly_is_a_norm(f * fbar * (t - 1))
       False
       sage: poly_is_a_norm(4*t**2 + (z**3 + z**2 + 5)*t + 4)
       False
    """
    factors = dict(g.factor())
    for h in factors:
        assert h.is_monic()
        hbar = poly_involution(h)
        hbar = hbar / hbar.leading_coefficient()
        if hbar == h and factors[h] % 2 != 0:
            return False
        elif factors[h] != factors[hbar]:
            return False
    return True