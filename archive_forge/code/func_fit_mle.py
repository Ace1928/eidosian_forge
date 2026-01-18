import numpy as np
from scipy import special, stats
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.tsa.arma_mle import Arma
def fit_mle(self, order, start_params=None, method='nm', maxiter=5000, tol=1e-08, **kwds):
    nar, nma = order
    if start_params is not None:
        if len(start_params) != nar + nma + 2:
            raise ValueError('start_param need sum(order) + 2 elements')
    else:
        start_params = np.concatenate((0.05 * np.ones(nar + nma), [5, 1]))
    res = super().fit_mle(order=order, start_params=start_params, method=method, maxiter=maxiter, tol=tol, **kwds)
    return res