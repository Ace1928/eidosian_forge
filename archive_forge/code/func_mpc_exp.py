import sys
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, BACKEND
from .libmpf import (\
from .libelefun import (\
def mpc_exp(z, prec, rnd=round_fast):
    """
    Complex exponential function.

    We use the direct formula exp(a+bi) = exp(a) * (cos(b) + sin(b)*i)
    for the computation. This formula is very nice because it is
    pefectly stable; since we just do real multiplications, the only
    numerical errors that can creep in are single-ulp rounding errors.

    The formula is efficient since mpmath's real exp is quite fast and
    since we can compute cos and sin simultaneously.

    It is no problem if a and b are large; if the implementations of
    exp/cos/sin are accurate and efficient for all real numbers, then
    so is this function for all complex numbers.
    """
    a, b = z
    if a == fzero:
        return mpf_cos_sin(b, prec, rnd)
    if b == fzero:
        return (mpf_exp(a, prec, rnd), fzero)
    mag = mpf_exp(a, prec + 4, rnd)
    c, s = mpf_cos_sin(b, prec + 4, rnd)
    re = mpf_mul(mag, c, prec, rnd)
    im = mpf_mul(mag, s, prec, rnd)
    return (re, im)