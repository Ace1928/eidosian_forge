import numpy as np
from statsmodels.genmod.bayes_mixed_glm import (BinomialBayesMixedGLM,
import pandas as pd
from scipy import sparse
from numpy.testing import assert_allclose, assert_equal
from scipy.optimize import approx_fprime
def gen_crossed_logit_pandas(nc, cs, s1, s2):
    np.random.seed(3799)
    a = np.kron(np.arange(nc), np.ones(cs))
    b = np.kron(np.ones(cs), np.arange(nc))
    fe = np.ones(nc * cs)
    vc = np.zeros(nc * cs)
    for i in np.unique(a):
        ii = np.flatnonzero(a == i)
        vc[ii] += s1 * np.random.normal()
    for i in np.unique(b):
        ii = np.flatnonzero(b == i)
        vc[ii] += s2 * np.random.normal()
    lp = -0.5 * fe + vc
    pr = 1 / (1 + np.exp(-lp))
    y = 1 * (np.random.uniform(size=nc * cs) < pr)
    ident = np.zeros(2 * nc, dtype=int)
    ident[nc:] = 1
    df = pd.DataFrame({'fe': fe, 'a': a, 'b': b, 'y': y})
    return df