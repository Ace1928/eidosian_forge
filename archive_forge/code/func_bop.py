from numba import jit, njit
from numba.core import types
from numba.core.extending import overload
def bop():
    return _GLOBAL1 + a - b