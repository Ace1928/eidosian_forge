import numpy as np
from scipy.optimize import minimize
import GPy
from GPy.kern import Kern
from GPy.core import Param
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
def optimize_acq(func, m, m1, fixed, num_f):
    """Optimize acquisition function."""
    opts = {'maxiter': 200, 'maxfun': 200, 'disp': False}
    T = 10
    best_value = -999
    best_theta = m1.X[0, :]
    bounds = [(0, 1) for _ in range(m.X.shape[1] - num_f)]
    for ii in range(T):
        x0 = np.random.uniform(0, 1, m.X.shape[1] - num_f)
        res = minimize(lambda x: -func(m, m1, x, fixed), x0, bounds=bounds, method='L-BFGS-B', options=opts)
        val = func(m, m1, res.x, fixed)
        if val > best_value:
            best_value = val
            best_theta = res.x
    return np.clip(best_theta, 0, 1)