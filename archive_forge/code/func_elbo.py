import numpy as np
from statsmodels.genmod.bayes_mixed_glm import (BinomialBayesMixedGLM,
import pandas as pd
from scipy import sparse
from numpy.testing import assert_allclose, assert_equal
from scipy.optimize import approx_fprime
def elbo(vec):
    n = len(vec) // 2
    return glmm1.vb_elbo(vec[:n], vec[n:])