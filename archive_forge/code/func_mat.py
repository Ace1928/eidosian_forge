import numpy as np
from collections import defaultdict
import statsmodels.base.model as base
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import links
from statsmodels.genmod.families import varfuncs
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
from statsmodels.tools.decorators import cache_readonly
def mat(self, dim, term):
    if dim < 3:
        msg = 'Groups must have size at least 3 for ' + 'autoregressive covariance.'
        raise ValueError(msg)
    if term == 0:
        return np.eye(dim)
    elif term == 1:
        mat = np.zeros((dim, dim))
        mat.flat[1::dim + 1] = 1
        mat += mat.T
        return mat
    elif term == 2:
        mat = np.zeros((dim, dim))
        mat[0, 0] = 1
        mat[dim - 1, dim - 1] = 1
        return mat
    else:
        return None