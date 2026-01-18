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
def cov_var_repr(self):
    """
        Gives the covariance matrix of the corresponding VAR-representation.

        More precisely, the covariance matrix of the vector consisting of the
        columns of the corresponding VAR coefficient matrices (i.e.
        vec(self.var_rep)).

        Returns
        -------
        cov : array (neqs**2 * k_ar x neqs**2 * k_ar)
        """
    if self.k_ar - 1 == 0:
        return self.cov_params_wo_det
    vecm_var_transformation = np.zeros((self.neqs ** 2 * self.k_ar, self.neqs ** 2 * self.k_ar))
    eye = np.identity(self.neqs ** 2)
    vecm_var_transformation[:self.neqs ** 2, :2 * self.neqs ** 2] = hstack((eye, eye))
    for i in range(2, self.k_ar):
        start_row = self.neqs ** 2 + (i - 2) * self.neqs ** 2
        start_col = self.neqs ** 2 + (i - 2) * self.neqs ** 2
        vecm_var_transformation[start_row:start_row + self.neqs ** 2, start_col:start_col + 2 * self.neqs ** 2] = hstack((-eye, eye))
    vecm_var_transformation[-self.neqs ** 2:, -self.neqs ** 2:] = -eye
    vvt = vecm_var_transformation
    return vvt @ self.cov_params_wo_det @ vvt.T