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
def stderr_coint(self):
    """
        Standard errors of beta and deterministic terms inside the
        cointegration relation.

        Notes
        -----
        See p. 297 in [1]_. Using the rule

        .. math::

           vec(B R) = (B' \\otimes I) vec(R)

        for two matrices B and R which are compatible for multiplication.
        This is rule (3) on p. 662 in [1]_.

        References
        ----------
        .. [1] Lütkepohl, H. 2005. *New Introduction to Multiple Time Series Analysis*. Springer.
        """
    r = self.coint_rank
    _, r1 = _r_matrices(self._delta_y_1_T, self._y_lag1, self._delta_x)
    r12 = r1[r:]
    if r12.size == 0:
        return np.zeros((r, r))
    mat1 = inv(r12.dot(r12.T))
    mat1 = np.kron(mat1.T, np.identity(r))
    det = self.det_coef_coint.shape[0]
    mat2 = np.kron(np.identity(self.neqs - r + det), inv(self.alpha.T @ inv(self.sigma_u) @ self.alpha))
    first_rows = np.zeros((r, r))
    last_rows_1d = np.sqrt(np.diag(mat1.dot(mat2)))
    last_rows = last_rows_1d.reshape((self.neqs - r + det, r), order='F')
    return vstack((first_rows, last_rows))