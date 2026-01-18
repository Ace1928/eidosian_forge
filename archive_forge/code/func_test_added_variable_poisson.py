import numpy as np
from numpy.testing import assert_array_less, assert_equal, assert_raises
from pandas import DataFrame, Series
import pytest
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import (
@pytest.mark.matplotlib
def test_added_variable_poisson(self, close_figures):
    np.random.seed(3446)
    n = 100
    p = 3
    exog = np.random.normal(size=(n, p))
    lin_pred = 4 + exog[:, 0] + 0.2 * exog[:, 1] ** 2
    expval = np.exp(lin_pred)
    endog = np.random.poisson(expval)
    model = sm.GLM(endog, exog, family=sm.families.Poisson())
    results = model.fit()
    for focus_col in (0, 1, 2):
        for use_glm_weights in (False, True):
            for resid_type in ('resid_deviance', 'resid_response'):
                weight_str = ['Unweighted', 'Weighted'][use_glm_weights]
                for j in (0, 1):
                    if j == 0:
                        fig = plot_added_variable(results, focus_col, use_glm_weights=use_glm_weights, resid_type=resid_type)
                        ti = 'Added variable plot'
                    else:
                        fig = results.plot_added_variable(focus_col, use_glm_weights=use_glm_weights, resid_type=resid_type)
                        ti = 'Added variable plot (called as method)'
                    ax = fig.get_axes()[0]
                    add_lowess(ax)
                    ax.set_position([0.1, 0.1, 0.8, 0.7])
                    effect_str = ['Linear effect, slope=1', 'Quadratic effect', 'No effect'][focus_col]
                    ti += '\nPoisson regression\n'
                    ti += effect_str + '\n'
                    ti += weight_str + '\n'
                    ti += "Using '%s' residuals" % resid_type
                    ax.set_title(ti)
                    close_or_save(pdf, fig)
                    close_figures()