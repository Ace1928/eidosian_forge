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
@Appender(_plot_leverage_resid2_doc.format({'extra_params_doc': 'results : object\n    Results for a fitted regression model'}))
def plot_leverage_resid2(results, alpha=0.05, ax=None, **kwargs):
    infl = results.get_influence()
    return _plot_leverage_resid2(results, infl, alpha=alpha, ax=ax, **kwargs)