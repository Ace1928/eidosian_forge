import numpy as np
from scipy import special
from scipy.optimize import OptimizeResult
from scipy.optimize._zeros_py import (  # noqa: F401
def post_termination_check(work):
    work.n += 1
    work.Sk = np.concatenate((work.Sk, work.Sn[:, np.newaxis]), axis=-1)
    return