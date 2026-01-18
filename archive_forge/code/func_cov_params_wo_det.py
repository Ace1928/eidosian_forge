from collections import defaultdict
import numpy as np
from numpy import hstack, vstack
from numpy.linalg import inv, svd
import scipy
import scipy.stats
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.validation import string_like
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.coint_tables import c_sja, c_sjt
from statsmodels.tsa.tsatools import duplication_matrix, lagmat, vec
from statsmodels.tsa.vector_ar.hypothesis_test_results import (
import statsmodels.tsa.vector_ar.irf as irf
import statsmodels.tsa.vector_ar.plotting as plot
from statsmodels.tsa.vector_ar.util import get_index, seasonal_dummies
from statsmodels.tsa.vector_ar.var_model import (
@cache_readonly
def cov_params_wo_det(self):
    start_i = self.neqs ** 2
    end_i = start_i + self.neqs * self.det_coef_coint.shape[0]
    to_drop_i = np.arange(start_i, end_i)
    cov = self.cov_params_default
    cov_size = len(cov)
    to_drop_o = np.arange(cov_size - self.det_coef.size, cov_size)
    to_drop = np.union1d(to_drop_i, to_drop_o)
    mask = np.ones(cov.shape, dtype=bool)
    mask[to_drop] = False
    mask[:, to_drop] = False
    cov_size_new = mask.sum(axis=0)[0]
    return cov[mask].reshape((cov_size_new, cov_size_new))