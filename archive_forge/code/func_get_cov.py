import numpy as np
import pandas as pd
import patsy
import statsmodels.base.model as base
from statsmodels.regression.linear_model import OLS
import collections
from scipy.optimize import minimize
from statsmodels.iolib import summary2
from statsmodels.tools.numdiff import approx_fprime
import warnings
def get_cov(self, time, sc, sm):
    da = np.subtract.outer(time, time)
    ds = np.add.outer(sm, sm) / 2
    qmat = da * da / ds
    cm = np.exp(-qmat / 2) / np.sqrt(ds)
    cm *= np.outer(sm, sm) ** 0.25
    cm *= np.outer(sc, sc)
    return cm