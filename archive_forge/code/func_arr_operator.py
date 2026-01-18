from __future__ import annotations
import numba as nb
import numpy as np
import os
def arr_operator(f):
    """Define and register a new array composite operator"""
    if jit_enabled:
        f2 = nb.vectorize(f)
        f2._compile_for_argtys((nb.types.int32, nb.types.int32))
        f2._compile_for_argtys((nb.types.int64, nb.types.int64))
        f2._compile_for_argtys((nb.types.float32, nb.types.float32))
        f2._compile_for_argtys((nb.types.float64, nb.types.float64))
        f2._frozen = True
    else:
        f2 = np.vectorize(f)
    composite_op_lookup[f.__name__] = f2
    return f2