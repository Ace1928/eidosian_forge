import pandas as pd
import patsy
import numpy as np
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.compat.pandas import Appender
@Appender(_predict_functional_doc)
def predict_functional(result, focus_var, summaries=None, values=None, summaries2=None, values2=None, alpha=0.05, ci_method='pointwise', linear=True, num_points=10, exog=None, exog2=None, **kwargs):
    if ci_method not in ('pointwise', 'scheffe', 'simultaneous'):
        raise ValueError('confidence band method must be one of `pointwise`, `scheffe`, and `simultaneous`.')
    contrast = values2 is not None or summaries2 is not None
    if contrast and (not linear):
        raise ValueError('`linear` must be True for computing contrasts')
    model = result.model
    if exog is not None:
        if any((x is not None for x in [summaries, summaries2, values, values2])):
            raise ValueError('if `exog` is provided then do not provide `summaries` or `values`')
        fexog = exog
        dexog = patsy.dmatrix(model.data.design_info, fexog, return_type='dataframe')
        fvals = exog[focus_var]
        if exog2 is not None:
            fexog2 = exog
            dexog2 = patsy.dmatrix(model.data.design_info, fexog2, return_type='dataframe')
            fvals2 = fvals
    else:
        values, summaries, values2, summaries2 = _check_args(values, summaries, values2, summaries2)
        dexog, fexog, fvals = _make_exog(result, focus_var, summaries, values, num_points)
        if len(summaries2) + len(values2) > 0:
            dexog2, fexog2, fvals2 = _make_exog(result, focus_var, summaries2, values2, num_points)
    from statsmodels.genmod.generalized_linear_model import GLM
    from statsmodels.genmod.generalized_estimating_equations import GEE
    if isinstance(result.model, (GLM, GEE)):
        kwargs_pred = kwargs.copy()
        kwargs_pred.update({'which': 'linear'})
    else:
        kwargs_pred = kwargs
    pred = result.predict(exog=fexog, **kwargs_pred)
    if contrast:
        pred2 = result.predict(exog=fexog2, **kwargs_pred)
        pred = pred - pred2
        dexog = dexog - dexog2
    if ci_method == 'pointwise':
        t_test = result.t_test(dexog)
        cb = t_test.conf_int(alpha=alpha)
    elif ci_method == 'scheffe':
        t_test = result.t_test(dexog)
        sd = t_test.sd
        cb = np.zeros((num_points, 2))
        from scipy.stats.distributions import f as fdist
        df1 = result.model.exog.shape[1]
        df2 = result.model.exog.shape[0] - df1
        qf = fdist.cdf(1 - alpha, df1, df2)
        fx = sd * np.sqrt(df1 * qf)
        cb[:, 0] = pred - fx
        cb[:, 1] = pred + fx
    elif ci_method == 'simultaneous':
        sigma, c = _glm_basic_scr(result, dexog, alpha)
        cb = np.zeros((dexog.shape[0], 2))
        cb[:, 0] = pred - c * sigma
        cb[:, 1] = pred + c * sigma
    if not linear:
        link = result.family.link
        pred = link.inverse(pred)
        cb = link.inverse(cb)
    return (pred, cb, fvals)