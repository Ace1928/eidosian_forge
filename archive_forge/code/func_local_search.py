import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize, Bounds
from scipy.special import gammaln
from scipy._lib._util import check_random_state
from scipy.optimize._constraints import new_bounds_to_old
def local_search(self, x, e):
    x_tmp = np.copy(x)
    mres = self.minimizer(self.func_wrapper.fun, x, **self.kwargs)
    if 'njev' in mres:
        self.func_wrapper.ngev += mres.njev
    if 'nhev' in mres:
        self.func_wrapper.nhev += mres.nhev
    is_finite = np.all(np.isfinite(mres.x)) and np.isfinite(mres.fun)
    in_bounds = np.all(mres.x >= self.lower) and np.all(mres.x <= self.upper)
    is_valid = is_finite and in_bounds
    if is_valid and mres.fun < e:
        return (mres.fun, mres.x)
    else:
        return (e, x_tmp)