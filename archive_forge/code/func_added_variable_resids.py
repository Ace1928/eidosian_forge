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
def added_variable_resids(results, focus_exog, resid_type=None, use_glm_weights=True, fit_kwargs=None):
    """
    Residualize the endog variable and a 'focus' exog variable in a
    regression model with respect to the other exog variables.

    Parameters
    ----------
    results : regression results instance
        A fitted model including the focus exog and all other
        predictors of interest.
    focus_exog : {int, str}
        The column of results.model.exog or a variable name that is
        to be residualized against the other predictors.
    resid_type : str
        The type of residuals to use for the dependent variable.  If
        None, uses `resid_deviance` for GLM/GEE and `resid` otherwise.
    use_glm_weights : bool
        Only used if the model is a GLM or GEE.  If True, the
        residuals for the focus predictor are computed using WLS, with
        the weights obtained from the IRLS calculations for fitting
        the GLM.  If False, unweighted regression is used.
    fit_kwargs : dict, optional
        Keyword arguments to be passed to fit when refitting the
        model.

    Returns
    -------
    endog_resid : array_like
        The residuals for the original exog
    focus_exog_resid : array_like
        The residuals for the focus predictor

    Notes
    -----
    The 'focus variable' residuals are always obtained using linear
    regression.

    Currently only GLM, GEE, and OLS models are supported.
    """
    model = results.model
    if not isinstance(model, (GEE, GLM, OLS)):
        raise ValueError('model type %s not supported for added variable residuals' % model.__class__.__name__)
    exog = model.exog
    endog = model.endog
    focus_exog, focus_col = utils.maybe_name_or_idx(focus_exog, model)
    focus_exog_vals = exog[:, focus_col]
    if resid_type is None:
        if isinstance(model, (GEE, GLM)):
            resid_type = 'resid_deviance'
        else:
            resid_type = 'resid'
    ii = range(exog.shape[1])
    ii = list(ii)
    ii.pop(focus_col)
    reduced_exog = exog[:, ii]
    start_params = results.params[ii]
    klass = model.__class__
    kwargs = model._get_init_kwds()
    new_model = klass(endog, reduced_exog, **kwargs)
    args = {'start_params': start_params}
    if fit_kwargs is not None:
        args.update(fit_kwargs)
    new_result = new_model.fit(**args)
    if not getattr(new_result, 'converged', True):
        raise ValueError('fit did not converge when calculating added variable residuals')
    try:
        endog_resid = getattr(new_result, resid_type)
    except AttributeError:
        raise ValueError("'%s' residual type not available" % resid_type)
    import statsmodels.regression.linear_model as lm
    if isinstance(model, (GLM, GEE)) and use_glm_weights:
        weights = model.family.weights(results.fittedvalues)
        if hasattr(model, 'data_weights'):
            weights = weights * model.data_weights
        lm_results = lm.WLS(focus_exog_vals, reduced_exog, weights).fit()
    else:
        lm_results = lm.OLS(focus_exog_vals, reduced_exog).fit()
    focus_exog_resid = lm_results.resid
    return (endog_resid, focus_exog_resid)