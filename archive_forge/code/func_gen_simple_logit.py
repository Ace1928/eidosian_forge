import numpy as np
from statsmodels.genmod.bayes_mixed_glm import (BinomialBayesMixedGLM,
import pandas as pd
from scipy import sparse
from numpy.testing import assert_allclose, assert_equal
from scipy.optimize import approx_fprime
def gen_simple_logit(nc, cs, s):
    np.random.seed(3799)
    exog_vc = np.kron(np.eye(nc), np.ones((cs, 1)))
    exog_fe = np.random.normal(size=(nc * cs, 2))
    vc = s * np.random.normal(size=nc)
    lp = np.dot(exog_fe, np.r_[1, -1]) + np.dot(exog_vc, vc)
    pr = 1 / (1 + np.exp(-lp))
    y = 1 * (np.random.uniform(size=nc * cs) < pr)
    ident = np.zeros(nc, dtype=int)
    return (y, exog_fe, exog_vc, ident)