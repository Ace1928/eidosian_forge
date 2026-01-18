import common_z3 as CM_Z3
import ctypes
from .z3 import *
def model_str(m, as_str=True):
    """
    Returned a 'sorted' model (so that it's easier to see)
    The model is sorted by its key,
    e.g. if the model is y = 3 , x = 10, then the result is
    x = 10, y = 3

    EXAMPLES:
    see doctest examples from function prove()

    """
    if z3_debug():
        assert m is None or m == [] or isinstance(m, ModelRef)
    if m:
        vs = [(v, m[v]) for v in m]
        vs = sorted(vs, key=lambda a, _: str(a))
        if as_str:
            return '\n'.join(['{} = {}'.format(k, v) for k, v in vs])
        else:
            return vs
    else:
        return str(m) if as_str else m