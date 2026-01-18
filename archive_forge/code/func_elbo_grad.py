from a common formula are constrained to have the same standard
import numpy as np
from scipy.optimize import minimize
from scipy import sparse
import statsmodels.base.model as base
from statsmodels.iolib import summary2
from statsmodels.genmod import families
import pandas as pd
import warnings
import patsy
def elbo_grad(x):
    n = len(x) // 2
    gm, gs = self.vb_elbo_grad(x[:n], np.exp(x[n:]))
    gs *= np.exp(x[n:])
    return -np.concatenate((gm, gs))