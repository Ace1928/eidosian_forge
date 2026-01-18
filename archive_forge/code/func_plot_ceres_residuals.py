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
@Appender(_plot_ceres_residuals_doc % {'extra_params_doc': 'results : Results\n        Results instance of a fitted regression model.'})
def plot_ceres_residuals(results, focus_exog, frac=0.66, cond_means=None, ax=None):
    model = results.model
    focus_exog, focus_col = utils.maybe_name_or_idx(focus_exog, model)
    presid = ceres_resids(results, focus_exog, frac=frac, cond_means=cond_means)
    focus_exog_vals = model.exog[:, focus_col]
    fig, ax = utils.create_mpl_ax(ax)
    ax.plot(focus_exog_vals, presid, 'o', alpha=0.6)
    ax.set_title('CERES residuals plot', fontsize='large')
    ax.set_xlabel(focus_exog, size=15)
    ax.set_ylabel('Component plus residual', size=15)
    return fig