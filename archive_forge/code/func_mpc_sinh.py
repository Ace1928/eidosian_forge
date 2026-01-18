import sys
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, BACKEND
from .libmpf import (\
from .libelefun import (\
def mpc_sinh(z, prec, rnd=round_fast):
    """Complex hyperbolic sine. Computed as sinh(z) = -i*sin(z*i)."""
    a, b = z
    b, a = mpc_sin((b, a), prec, rnd)
    return (a, b)