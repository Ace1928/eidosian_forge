import warnings
import numpy as np
import pandas as pd
import patsy
from scipy import sparse
from scipy.stats.distributions import norm
from statsmodels.base._penalties import Penalty
import statsmodels.base.model as base
from statsmodels.tools import data as data_tools
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
def score_full(self, params, calc_fe):
    """
        Returns the score with respect to untransformed parameters.

        Calculates the score vector for the profiled log-likelihood of
        the mixed effects model with respect to the parameterization
        in which the random effects covariance matrix is represented
        in its full form (not using the Cholesky factor).

        Parameters
        ----------
        params : MixedLMParams or array_like
            The parameter at which the score function is evaluated.
            If array-like, must contain the packed random effects
            parameters (cov_re and vcomp) without fe_params.
        calc_fe : bool
            If True, calculate the score vector for the fixed effects
            parameters.  If False, this vector is not calculated, and
            a vector of zeros is returned in its place.

        Returns
        -------
        score_fe : array_like
            The score vector with respect to the fixed effects
            parameters.
        score_re : array_like
            The score vector with respect to the random effects
            parameters (excluding variance components parameters).
        score_vc : array_like
            The score vector with respect to variance components
            parameters.

        Notes
        -----
        `score_re` is taken with respect to the parameterization in
        which `cov_re` is represented through its lower triangle
        (without taking the Cholesky square root).
        """
    fe_params = params.fe_params
    cov_re = params.cov_re
    vcomp = params.vcomp
    try:
        cov_re_inv = np.linalg.inv(cov_re)
    except np.linalg.LinAlgError:
        cov_re_inv = np.linalg.pinv(cov_re)
        self._cov_sing += 1
    score_fe = np.zeros(self.k_fe)
    score_re = np.zeros(self.k_re2)
    score_vc = np.zeros(self.k_vc)
    if self.cov_pen is not None:
        score_re -= self.cov_pen.deriv(cov_re, cov_re_inv)
    if calc_fe and self.fe_pen is not None:
        score_fe -= self.fe_pen.deriv(fe_params)
    rvir = 0.0
    xtvir = 0.0
    xtvix = 0.0
    xtax = [0.0] * (self.k_re2 + self.k_vc)
    dlv = np.zeros(self.k_re2 + self.k_vc)
    rvavr = np.zeros(self.k_re2 + self.k_vc)
    for group_ix, group in enumerate(self.group_labels):
        vc_var = self._expand_vcomp(vcomp, group_ix)
        exog = self.exog_li[group_ix]
        ex_r, ex2_r = (self._aex_r[group_ix], self._aex_r2[group_ix])
        solver = _smw_solver(1.0, ex_r, ex2_r, cov_re_inv, 1 / vc_var)
        resid = self.endog_li[group_ix]
        if self.k_fe > 0:
            expval = np.dot(exog, fe_params)
            resid = resid - expval
        if self.reml:
            viexog = solver(exog)
            xtvix += np.dot(exog.T, viexog)
        vir = solver(resid)
        for jj, matl, matr, vsl, vsr, sym in self._gen_dV_dPar(ex_r, solver, group_ix):
            dlv[jj] = _dotsum(matr, vsl)
            if not sym:
                dlv[jj] += _dotsum(matl, vsr)
            ul = _dot(vir, matl)
            ur = ul.T if sym else _dot(matr.T, vir)
            ulr = np.dot(ul, ur)
            rvavr[jj] += ulr
            if not sym:
                rvavr[jj] += ulr.T
            if self.reml:
                ul = _dot(viexog.T, matl)
                ur = ul.T if sym else _dot(matr.T, viexog)
                ulr = np.dot(ul, ur)
                xtax[jj] += ulr
                if not sym:
                    xtax[jj] += ulr.T
        if self.k_re > 0:
            score_re -= 0.5 * dlv[0:self.k_re2]
        if self.k_vc > 0:
            score_vc -= 0.5 * dlv[self.k_re2:]
        rvir += np.dot(resid, vir)
        if calc_fe:
            xtvir += np.dot(exog.T, vir)
    fac = self.n_totobs
    if self.reml:
        fac -= self.k_fe
    if calc_fe and self.k_fe > 0:
        score_fe += fac * xtvir / rvir
    if self.k_re > 0:
        score_re += 0.5 * fac * rvavr[0:self.k_re2] / rvir
    if self.k_vc > 0:
        score_vc += 0.5 * fac * rvavr[self.k_re2:] / rvir
    if self.reml:
        xtvixi = np.linalg.inv(xtvix)
        for j in range(self.k_re2):
            score_re[j] += 0.5 * _dotsum(xtvixi.T, xtax[j])
        for j in range(self.k_vc):
            score_vc[j] += 0.5 * _dotsum(xtvixi.T, xtax[self.k_re2 + j])
    return (score_fe, score_re, score_vc)