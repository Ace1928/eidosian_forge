from numba import jit, njit
from numba.core import types
from numba.core.extending import overload
@overload(baz, inline='always')
def baz_ol():

    def impl():
        return _GLOBAL1 + 10
    return impl