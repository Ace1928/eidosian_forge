import sys
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, BACKEND
from .libmpf import (\
from .libelefun import (\
def mpc_arg(z, prec, rnd=round_fast):
    """Argument of a complex number. Returns an mpf value."""
    a, b = z
    return mpf_atan2(b, a, prec, rnd)