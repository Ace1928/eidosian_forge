from numba import jit, njit
from numba.core import types
from numba.core.extending import overload
def outer_multiple(a):
    return inner(a) * more(a)