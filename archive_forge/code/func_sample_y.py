import warnings
from numbers import Integral, Real
from operator import itemgetter
import numpy as np
import scipy.optimize
from scipy.linalg import cho_solve, cholesky, solve_triangular
from ..base import BaseEstimator, MultiOutputMixin, RegressorMixin, _fit_context, clone
from ..preprocessing._data import _handle_zeros_in_scale
from ..utils import check_random_state
from ..utils._param_validation import Interval, StrOptions
from ..utils.optimize import _check_optimize_result
from .kernels import RBF, Kernel
from .kernels import ConstantKernel as C
def sample_y(self, X, n_samples=1, random_state=0):
    """Draw samples from Gaussian process and evaluate at X.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Query points where the GP is evaluated.

        n_samples : int, default=1
            Number of samples drawn from the Gaussian process per query point.

        random_state : int, RandomState instance or None, default=0
            Determines random number generation to randomly draw samples.
            Pass an int for reproducible results across multiple function
            calls.
            See :term:`Glossary <random_state>`.

        Returns
        -------
        y_samples : ndarray of shape (n_samples_X, n_samples), or             (n_samples_X, n_targets, n_samples)
            Values of n_samples samples drawn from Gaussian process and
            evaluated at query points.
        """
    rng = check_random_state(random_state)
    y_mean, y_cov = self.predict(X, return_cov=True)
    if y_mean.ndim == 1:
        y_samples = rng.multivariate_normal(y_mean, y_cov, n_samples).T
    else:
        y_samples = [rng.multivariate_normal(y_mean[:, target], y_cov[..., target], n_samples).T[:, np.newaxis] for target in range(y_mean.shape[1])]
        y_samples = np.hstack(y_samples)
    return y_samples