import numpy as np
from scipy.optimize import minimize, Bounds
def test_gh12922():

    def objective(x):
        return np.array([np.sum((x + 1) ** 4)])
    cons = {'type': 'ineq', 'fun': lambda x: -x[0] ** 2}
    n = 25
    x0 = np.linspace(-5, 5, n)
    opts = {'maxiter': 1000, 'verbose': 2}
    minimize(objective, x0=x0, method='trust-constr', constraints=cons, options=opts)
    opts = {'maxiter': 1000, 'verbose': 3}
    minimize(objective, x0=x0, method='trust-constr', constraints=cons, options=opts)