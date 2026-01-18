import warnings
from statsmodels.compat.pandas import Appender
from statsmodels.compat.python import lzip
from collections import defaultdict
import numpy as np
from statsmodels.graphics._regressionplots_doc import _plot_influence_doc
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import maybe_unwrap_results
def reset_ramsey(res, degree=5):
    """Ramsey's RESET specification test for linear models

    This is a general specification test, for additional non-linear effects
    in a model.

    Parameters
    ----------
    degree : int
        Maximum power to include in the RESET test.  Powers 0 and 1 are
        excluded, so that degree tests powers 2, ..., degree of the fitted
        values.

    Notes
    -----
    The test fits an auxiliary OLS regression where the design matrix, exog,
    is augmented by powers 2 to degree of the fitted values. Then it performs
    an F-test whether these additional terms are significant.

    If the p-value of the f-test is below a threshold, e.g. 0.1, then this
    indicates that there might be additional non-linear effects in the model
    and that the linear model is mis-specified.

    References
    ----------
    https://en.wikipedia.org/wiki/Ramsey_RESET_test
    """
    order = degree + 1
    k_vars = res.model.exog.shape[1]
    norm_values = np.asarray(res.fittedvalues)
    norm_values = norm_values / np.sqrt((norm_values ** 2).mean())
    y_fitted_vander = np.vander(norm_values, order)[:, :-2]
    exog = np.column_stack((res.model.exog, y_fitted_vander))
    exog /= np.sqrt((exog ** 2).mean(0))
    endog = res.model.endog / (res.model.endog ** 2).mean()
    res_aux = OLS(endog, exog).fit()
    r_matrix = np.eye(degree - 1, exog.shape[1], k_vars)
    return res_aux.f_test(r_matrix)