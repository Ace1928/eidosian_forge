from statsmodels.compat.pandas import Appender
from statsmodels.compat.python import lrange, lzip
import numpy as np
import pandas as pd
from patsy import dmatrix
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.graphics import utils
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.regression.linear_model import GLS, OLS, WLS
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.tools.tools import maybe_unwrap_results
from ._regressionplots_doc import (
def ceres_resids(results, focus_exog, frac=0.66, cond_means=None):
    """
    Calculate the CERES residuals (Conditional Expectation Partial
    Residuals) for a fitted model.

    Parameters
    ----------
    results : model results instance
        The fitted model for which the CERES residuals are calculated.
    focus_exog : int
        The column of results.model.exog used as the 'focus variable'.
    frac : float, optional
        Lowess smoothing parameter for estimating the conditional
        means.  Not used if `cond_means` is provided.
    cond_means : array_like, optional
        If provided, the columns of this array are the conditional
        means E[exog | focus exog], where exog ranges over some
        or all of the columns of exog other than focus exog.  If
        this is an empty nx0 array, the conditional means are
        treated as being zero.  If None, the conditional means are
        estimated.

    Returns
    -------
    An array containing the CERES residuals.

    Notes
    -----
    If `cond_means` is not provided, it is obtained by smoothing each
    column of exog (except the focus column) against the focus column.

    Currently only supports GLM, GEE, and OLS models.
    """
    model = results.model
    if not isinstance(model, (GLM, GEE, OLS)):
        raise ValueError('ceres residuals not available for %s' % model.__class__.__name__)
    focus_exog, focus_col = utils.maybe_name_or_idx(focus_exog, model)
    ix_nf = range(len(results.params))
    ix_nf = list(ix_nf)
    ix_nf.pop(focus_col)
    nnf = len(ix_nf)
    if cond_means is None:
        pexog = model.exog[:, ix_nf]
        pexog -= pexog.mean(0)
        u, s, vt = np.linalg.svd(pexog, 0)
        ii = np.flatnonzero(s > 1e-06)
        pexog = u[:, ii]
        fcol = model.exog[:, focus_col]
        cond_means = np.empty((len(fcol), pexog.shape[1]))
        for j in range(pexog.shape[1]):
            y0 = pexog[:, j]
            cf = lowess(y0, fcol, frac=frac, return_sorted=False)
            cond_means[:, j] = cf
    new_exog = np.concatenate((model.exog[:, ix_nf], cond_means), axis=1)
    klass = model.__class__
    init_kwargs = model._get_init_kwds()
    new_model = klass(model.endog, new_exog, **init_kwargs)
    new_result = new_model.fit()
    presid = model.endog - new_result.fittedvalues
    if isinstance(model, (GLM, GEE)):
        presid *= model.family.link.deriv(new_result.fittedvalues)
    if new_exog.shape[1] > nnf:
        presid += np.dot(new_exog[:, nnf:], new_result.params[nnf:])
    return presid