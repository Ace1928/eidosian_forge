import numpy as np
from scipy import stats, signal
import matplotlib.pyplot as plt
def fitls(self, data, dt):
    """assumes data is 1d, univariate time series
        formula from sitmo
        """
    nobs = len(data) - 1
    exog = np.column_stack((np.ones(nobs), np.log(data[:-1])))
    parest, res, rank, sing = np.linalg.lstsq(exog, np.log(data[1:]), rcond=-1)
    const, slope = parest
    errvar = res / (nobs - 2.0)
    kappa = -np.log(slope) / dt
    sigma = np.sqrt(errvar * kappa / (1 - np.exp(-2 * kappa * dt)))
    mu = const / (1 - np.exp(-kappa * dt)) + sigma ** 2 / 2.0 / kappa
    if np.shape(mu) == (1,):
        mu = mu[0]
    if np.shape(sigma) == (1,):
        sigma = sigma[0]
    return (mu, kappa, sigma)