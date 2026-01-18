import numpy as np
from statsmodels.genmod.bayes_mixed_glm import (BinomialBayesMixedGLM,
import pandas as pd
from scipy import sparse
from numpy.testing import assert_allclose, assert_equal
from scipy.optimize import approx_fprime
def gen_simple_poisson(nc, cs, s):
    np.random.seed(3799)
    exog_vc = np.kron(np.eye(nc), np.ones((cs, 1)))
    exog_fe = np.random.normal(size=(nc * cs, 2))
    vc = s * np.random.normal(size=nc)
    lp = np.dot(exog_fe, np.r_[0.1, -0.1]) + np.dot(exog_vc, vc)
    r = np.exp(lp)
    y = np.random.poisson(r)
    ident = np.zeros(nc, dtype=int)
    return (y, exog_fe, exog_vc, ident)