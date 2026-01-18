import numpy as np
from scipy import special
from statsmodels.sandbox.distributions.multivariate import mvstdtprob
from .extras import mvnormcdf
def quad2d(func=lambda x: 1, lower=(-10, -10), upper=(10, 10)):

    def fun(x, y):
        x = np.column_stack((x, y))
        return func(x)
    from scipy.integrate import dblquad
    return dblquad(fun, lower[0], upper[0], lambda y: lower[1], lambda y: upper[1])