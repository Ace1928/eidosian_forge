import numpy as np
from scipy import special
from scipy.optimize import OptimizeResult
from scipy.optimize._zeros_py import (  # noqa: F401
def pre_func_eval(work):
    work.h = h0 / 2 ** work.n
    xjc, wj = _get_pairs(work.n, h0, dtype=work.dtype, inclusive=work.n == minlevel)
    work.xj, work.wj = _transform_to_limits(xjc, wj, work.a, work.b)
    xj = work.xj.copy()
    xj[work.abinf] = xj[work.abinf] / (1 - xj[work.abinf] ** 2)
    xj[work.binf] = 1 / xj[work.binf] - 1 + work.a0[work.binf]
    xj[work.ainf] *= -1
    return xj