import itertools
from numbers import Integral, Real
import numpy as np
from scipy.special import gammainc
from ..base import BaseEstimator, _fit_context
from ..neighbors._base import VALID_METRICS
from ..utils import check_random_state
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import row_norms
from ..utils.validation import _check_sample_weight, check_is_fitted
from ._ball_tree import BallTree
from ._kd_tree import KDTree
def score_samples(self, X):
    """Compute the log-likelihood of each sample under the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            An array of points to query.  Last dimension should match dimension
            of training data (n_features).

        Returns
        -------
        density : ndarray of shape (n_samples,)
            Log-likelihood of each sample in `X`. These are normalized to be
            probability densities, so values will be low for high-dimensional
            data.
        """
    check_is_fitted(self)
    X = self._validate_data(X, order='C', dtype=np.float64, reset=False)
    if self.tree_.sample_weight is None:
        N = self.tree_.data.shape[0]
    else:
        N = self.tree_.sum_weight
    atol_N = self.atol * N
    log_density = self.tree_.kernel_density(X, h=self.bandwidth_, kernel=self.kernel, atol=atol_N, rtol=self.rtol, breadth_first=self.breadth_first, return_log=True)
    log_density -= np.log(N)
    return log_density