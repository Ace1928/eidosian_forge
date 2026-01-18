import copy
from textwrap import dedent
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from . import utils
from . import algorithms as algo
from .axisgrid import FacetGrid, _facet_docs
def fit_regression(self, ax=None, x_range=None, grid=None):
    """Fit the regression model."""
    self._check_statsmodels()
    if grid is None:
        if self.truncate:
            x_min, x_max = self.x_range
        elif ax is None:
            x_min, x_max = x_range
        else:
            x_min, x_max = ax.get_xlim()
        grid = np.linspace(x_min, x_max, 100)
    ci = self.ci
    if self.order > 1:
        yhat, yhat_boots = self.fit_poly(grid, self.order)
    elif self.logistic:
        from statsmodels.genmod.generalized_linear_model import GLM
        from statsmodels.genmod.families import Binomial
        yhat, yhat_boots = self.fit_statsmodels(grid, GLM, family=Binomial())
    elif self.lowess:
        ci = None
        grid, yhat = self.fit_lowess()
    elif self.robust:
        from statsmodels.robust.robust_linear_model import RLM
        yhat, yhat_boots = self.fit_statsmodels(grid, RLM)
    elif self.logx:
        yhat, yhat_boots = self.fit_logx(grid)
    else:
        yhat, yhat_boots = self.fit_fast(grid)
    if ci is None:
        err_bands = None
    else:
        err_bands = utils.ci(yhat_boots, ci, axis=0)
    return (grid, yhat, err_bands)