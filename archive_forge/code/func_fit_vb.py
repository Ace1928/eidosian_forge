from a common formula are constrained to have the same standard
import numpy as np
from scipy.optimize import minimize
from scipy import sparse
import statsmodels.base.model as base
from statsmodels.iolib import summary2
from statsmodels.genmod import families
import pandas as pd
import warnings
import patsy
def fit_vb(self, mean=None, sd=None, fit_method='BFGS', minim_opts=None, scale_fe=False, verbose=False):
    """
        Fit a model using the variational Bayes mean field approximation.

        Parameters
        ----------
        mean : array_like
            Starting value for VB mean vector
        sd : array_like
            Starting value for VB standard deviation vector
        fit_method : str
            Algorithm for scipy.minimize
        minim_opts : dict
            Options passed to scipy.minimize
        scale_fe : bool
            If true, the columns of the fixed effects design matrix
            are centered and scaled to unit variance before fitting
            the model.  The results are back-transformed so that the
            results are presented on the original scale.
        verbose : bool
            If True, print the gradient norm to the screen each time
            it is calculated.

        Notes
        -----
        The goal is to find a factored Gaussian approximation
        q1*q2*...  to the posterior distribution, approximately
        minimizing the KL divergence from the factored approximation
        to the actual posterior.  The KL divergence, or ELBO function
        has the form

            E* log p(y, fe, vcp, vc) - E* log q

        where E* is expectation with respect to the product of qj.

        References
        ----------
        Blei, Kucukelbir, McAuliffe (2017).  Variational Inference: A
        review for Statisticians
        https://arxiv.org/pdf/1601.00670.pdf
        """
    self.verbose = verbose
    if scale_fe:
        mn = self.exog.mean(0)
        sc = self.exog.std(0)
        self._exog_save = self.exog
        self.exog = self.exog.copy()
        ixs = np.flatnonzero(sc > 1e-08)
        self.exog[:, ixs] -= mn[ixs]
        self.exog[:, ixs] /= sc[ixs]
    n = self.k_fep + self.k_vcp + self.k_vc
    ml = self.k_fep + self.k_vcp + self.k_vc
    if mean is None:
        m = np.zeros(n)
    else:
        if len(mean) != ml:
            raise ValueError('mean has incorrect length, %d != %d' % (len(mean), ml))
        m = mean.copy()
    if sd is None:
        s = -0.5 + 0.1 * np.random.normal(size=n)
    else:
        if len(sd) != ml:
            raise ValueError('sd has incorrect length, %d != %d' % (len(sd), ml))
        s = np.log(sd)
    i1, i2 = (self.k_fep, self.k_fep + self.k_vcp)
    m[i1:i2] = np.where(m[i1:i2] < -1, -1, m[i1:i2])
    s = np.where(s < -1, -1, s)

    def elbo(x):
        n = len(x) // 2
        return -self.vb_elbo(x[:n], np.exp(x[n:]))

    def elbo_grad(x):
        n = len(x) // 2
        gm, gs = self.vb_elbo_grad(x[:n], np.exp(x[n:]))
        gs *= np.exp(x[n:])
        return -np.concatenate((gm, gs))
    start = np.concatenate((m, s))
    mm = minimize(elbo, start, jac=elbo_grad, method=fit_method, options=minim_opts)
    if not mm.success:
        warnings.warn('VB fitting did not converge')
    n = len(mm.x) // 2
    params = mm.x[0:n]
    va = np.exp(2 * mm.x[n:])
    if scale_fe:
        self.exog = self._exog_save
        del self._exog_save
        params[ixs] /= sc[ixs]
        va[ixs] /= sc[ixs] ** 2
    return BayesMixedGLMResults(self, params, va, mm)