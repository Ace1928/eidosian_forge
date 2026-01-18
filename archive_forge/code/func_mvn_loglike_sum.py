import numpy as np
from scipy import linalg
from scipy.linalg import toeplitz
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.datasets import sunspots
from statsmodels.tsa.arima_process import (
def mvn_loglike_sum(x, sigma):
    """loglike multivariate normal

    copied from GLS and adjusted names
    not sure why this differes from mvn_loglike
    """
    nobs = len(x)
    nobs2 = nobs / 2.0
    SSR = (x ** 2).sum()
    llf = -np.log(SSR) * nobs2
    llf -= (1 + np.log(np.pi / nobs2)) * nobs2
    if np.any(sigma) and sigma.ndim == 2:
        llf -= 0.5 * np.log(np.linalg.det(sigma))
    return llf