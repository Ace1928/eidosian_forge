from statsmodels.compat.pandas import Appender
from collections import defaultdict
import warnings
import numpy as np
import pandas as pd
from scipy import linalg as spl
from statsmodels.stats.correlation_tools import cov_nearest
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.validation import bool_like
def get_eyy(self, endog_expval, index):
    """
        Returns a matrix V such that V[i,j] is the joint probability
        that endog[i] = 1 and endog[j] = 1, based on the marginal
        probabilities of endog and the global odds ratio `current_or`.
        """
    current_or = self.dep_params
    ibd = self.ibd[index]
    if current_or == 1.0:
        vmat = np.outer(endog_expval, endog_expval)
    else:
        psum = endog_expval[:, None] + endog_expval[None, :]
        pprod = endog_expval[:, None] * endog_expval[None, :]
        pfac = np.sqrt((1.0 + psum * (current_or - 1.0)) ** 2 + 4 * current_or * (1.0 - current_or) * pprod)
        vmat = 1.0 + psum * (current_or - 1.0) - pfac
        vmat /= 2.0 * (current_or - 1)
    for bdl in ibd:
        evy = endog_expval[bdl[0]:bdl[1]]
        if self.endog_type == 'ordinal':
            vmat[bdl[0]:bdl[1], bdl[0]:bdl[1]] = np.minimum.outer(evy, evy)
        else:
            vmat[bdl[0]:bdl[1], bdl[0]:bdl[1]] = np.diag(evy)
    return vmat