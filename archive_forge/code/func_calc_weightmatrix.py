from statsmodels.compat.python import lrange
import numpy as np
from scipy import optimize, stats
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.base.model import (Model,
from statsmodels.regression.linear_model import (OLS, RegressionResults,
import statsmodels.stats.sandwich_covariance as smcov
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import _ensure_2d
def calc_weightmatrix(self, moms, weights_method='cov', wargs=(), params=None):
    """
        calculate omega or the weighting matrix

        Parameters
        ----------
        moms : ndarray
            moment conditions (nobs x nmoms) for all observations evaluated at
            a parameter value
        weights_method : str 'cov'
            If method='cov' is cov then the matrix is calculated as simple
            covariance of the moment conditions.
            see fit method for available aoptions for the weight and covariance
            matrix
        wargs : tuple or dict
            parameters that are required by some kernel methods to
            estimate the long-run covariance. Not used yet.

        Returns
        -------
        w : array (nmoms, nmoms)
            estimate for the weighting matrix or covariance of the moment
            condition


        Notes
        -----

        currently a constant cutoff window is used
        TODO: implement long-run cov estimators, kernel-based

        Newey-West
        Andrews
        Andrews-Moy????

        References
        ----------
        Greene
        Hansen, Bruce

        """
    nobs, k_moms = moms.shape
    if DEBUG:
        print(' momcov wargs', wargs)
    centered = not ('centered' in wargs and (not wargs['centered']))
    if not centered:
        moms_ = moms
    else:
        moms_ = moms - moms.mean()
    if weights_method == 'cov':
        w = np.dot(moms_.T, moms_)
        if 'ddof' in wargs:
            if wargs['ddof'] == 'k_params':
                w /= nobs - self.k_params
            else:
                if DEBUG:
                    print(' momcov ddof', wargs['ddof'])
                w /= nobs - wargs['ddof']
        else:
            w /= nobs
    elif weights_method == 'flatkernel':
        if 'maxlag' not in wargs:
            raise ValueError('flatkernel requires maxlag')
        maxlag = wargs['maxlag']
        h = np.ones(maxlag + 1)
        w = np.dot(moms_.T, moms_) / nobs
        for i in range(1, maxlag + 1):
            w += h[i] * np.dot(moms_[i:].T, moms_[:-i]) / (nobs - i)
    elif weights_method == 'hac':
        maxlag = wargs['maxlag']
        if 'kernel' in wargs:
            weights_func = wargs['kernel']
        else:
            weights_func = smcov.weights_bartlett
            wargs['kernel'] = weights_func
        w = smcov.S_hac_simple(moms_, nlags=maxlag, weights_func=weights_func)
        w /= nobs
    elif weights_method == 'iid':
        u = self.get_error(params)
        if centered:
            u -= u.mean(0)
        instrument = self.instrument
        w = np.dot(instrument.T, instrument).dot(np.dot(u.T, u)) / nobs
        if 'ddof' in wargs:
            if wargs['ddof'] == 'k_params':
                w /= nobs - self.k_params
            else:
                if DEBUG:
                    print(' momcov ddof', wargs['ddof'])
                w /= nobs - wargs['ddof']
        else:
            w /= nobs
    else:
        raise ValueError('weight method not available')
    return w