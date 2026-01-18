import warnings
from numbers import Integral, Real
import numpy as np
from scipy import linalg
from scipy.stats import chi2
from ..base import _fit_context
from ..utils import check_array, check_random_state
from ..utils._param_validation import Interval
from ..utils.extmath import fast_logdet
from ._empirical_covariance import EmpiricalCovariance, empirical_covariance
def reweight_covariance(self, data):
    """Re-weight raw Minimum Covariance Determinant estimates.

        Re-weight observations using Rousseeuw's method (equivalent to
        deleting outlying observations from the data set before
        computing location and covariance estimates) described
        in [RVDriessen]_.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            The data matrix, with p features and n samples.
            The data set must be the one which was used to compute
            the raw estimates.

        Returns
        -------
        location_reweighted : ndarray of shape (n_features,)
            Re-weighted robust location estimate.

        covariance_reweighted : ndarray of shape (n_features, n_features)
            Re-weighted robust covariance estimate.

        support_reweighted : ndarray of shape (n_samples,), dtype=bool
            A mask of the observations that have been used to compute
            the re-weighted robust location and covariance estimates.

        References
        ----------

        .. [RVDriessen] A Fast Algorithm for the Minimum Covariance
            Determinant Estimator, 1999, American Statistical Association
            and the American Society for Quality, TECHNOMETRICS
        """
    n_samples, n_features = data.shape
    mask = self.dist_ < chi2(n_features).isf(0.025)
    if self.assume_centered:
        location_reweighted = np.zeros(n_features)
    else:
        location_reweighted = data[mask].mean(0)
    covariance_reweighted = self._nonrobust_covariance(data[mask], assume_centered=self.assume_centered)
    support_reweighted = np.zeros(n_samples, dtype=bool)
    support_reweighted[mask] = True
    self._set_covariance(covariance_reweighted)
    self.location_ = location_reweighted
    self.support_ = support_reweighted
    X_centered = data - self.location_
    self.dist_ = np.sum(np.dot(X_centered, self.get_precision()) * X_centered, 1)
    return (location_reweighted, covariance_reweighted, support_reweighted)