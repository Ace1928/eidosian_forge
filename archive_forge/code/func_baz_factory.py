from numba import jit, njit
from numba.core import types
from numba.core.extending import overload
def baz_factory(a):
    b = 17 + a

    @njit(inline='always')
    def baz():
        return _GLOBAL1 + a - b
    return baz