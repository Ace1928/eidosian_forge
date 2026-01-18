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
@Appender(_plot_added_variable_doc % {'extra_params_doc': 'results : object\n    Results for a fitted regression model'})
def plot_added_variable(results, focus_exog, resid_type=None, use_glm_weights=True, fit_kwargs=None, ax=None):
    model = results.model
    fig, ax = utils.create_mpl_ax(ax)
    endog_resid, focus_exog_resid = added_variable_resids(results, focus_exog, resid_type=resid_type, use_glm_weights=use_glm_weights, fit_kwargs=fit_kwargs)
    ax.plot(focus_exog_resid, endog_resid, 'o', alpha=0.6)
    ax.set_title('Added variable plot', fontsize='large')
    if isinstance(focus_exog, str):
        xname = focus_exog
    else:
        xname = model.exog_names[focus_exog]
    ax.set_xlabel(xname, size=15)
    ax.set_ylabel(model.endog_names + ' residuals', size=15)
    return fig