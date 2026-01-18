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
def fit_poly(self, grid, order):
    """Regression using numpy polyfit for higher-order trends."""

    def reg_func(_x, _y):
        return np.polyval(np.polyfit(_x, _y, order), grid)
    x, y = (self.x, self.y)
    yhat = reg_func(x, y)
    if self.ci is None:
        return (yhat, None)
    yhat_boots = algo.bootstrap(x, y, func=reg_func, n_boot=self.n_boot, units=self.units, seed=self.seed)
    return (yhat, yhat_boots)