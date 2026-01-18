import numpy as np
from statsmodels.genmod.bayes_mixed_glm import (BinomialBayesMixedGLM,
import pandas as pd
from scipy import sparse
from numpy.testing import assert_allclose, assert_equal
from scipy.optimize import approx_fprime
def gen_crossed_logit(nc, cs, s1, s2):
    np.random.seed(3799)
    a = np.kron(np.eye(nc), np.ones((cs, 1)))
    b = np.kron(np.ones((cs, 1)), np.eye(nc))
    exog_vc = np.concatenate((a, b), axis=1)
    exog_fe = np.random.normal(size=(nc * cs, 1))
    vc = s1 * np.random.normal(size=2 * nc)
    vc[nc:] *= s2 / s1
    lp = np.dot(exog_fe, np.r_[-0.5]) + np.dot(exog_vc, vc)
    pr = 1 / (1 + np.exp(-lp))
    y = 1 * (np.random.uniform(size=nc * cs) < pr)
    ident = np.zeros(2 * nc, dtype=int)
    ident[nc:] = 1
    return (y, exog_fe, exog_vc, ident)