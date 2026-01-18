import numpy as np
from scipy import special
from scipy.optimize import OptimizeResult
from scipy.optimize._zeros_py import (  # noqa: F401
def customize_result(res, shape):
    if log and np.any(negative):
        pi = res['integral'].dtype.type(np.pi)
        j = np.complex64(1j)
        res['integral'] = res['integral'] + negative * pi * j
    else:
        res['integral'][negative] *= -1
    res['maxlevel'] = minlevel + res['nit'] - 1
    res['maxlevel'][res['nit'] == 0] = -1
    del res['nit']
    return shape