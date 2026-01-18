import numpy as np
from scipy.optimize import minimize, Bounds
def test_gh10880():
    bnds = Bounds(1, 2)
    opts = {'maxiter': 1000, 'verbose': 2}
    minimize(lambda x: x ** 2, x0=2.0, method='trust-constr', bounds=bnds, options=opts)
    opts = {'maxiter': 1000, 'verbose': 3}
    minimize(lambda x: x ** 2, x0=2.0, method='trust-constr', bounds=bnds, options=opts)