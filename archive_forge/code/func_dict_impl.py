from numba.core import types
from numba.core.imputils import lower_builtin
def dict_impl(iterable):
    res = Dict.empty(kt, vt)
    for k, v in iterable:
        res[k] = v
    return res