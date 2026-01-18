import numpy as np
from statsmodels.robust import mad
from scipy.optimize import minimize_scalar
def optim(lmbda):
    y, lmbda = self.transform_boxcox(x, lmbda)
    return (1 - lmbda) * sum_x + nobs / 2.0 * np.log(np.var(y))