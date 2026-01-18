import sys
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, BACKEND
from .libmpf import (\
from .libelefun import (\
def mpc_is_infnan(z):
    """Check if either real or imaginary part is infinite or nan"""
    re, im = z
    if re in _infs_nan:
        return True
    if im in _infs_nan:
        return True
    return False