import warnings
from scipy import linalg, special
from scipy._lib._util import check_random_state
from numpy import (asarray, atleast_2d, reshape, zeros, newaxis, exp, pi,
import numpy as np
from . import _mvn
from ._stats import gaussian_kernel_estimate, gaussian_kernel_estimate_log
from scipy.special import logsumexp  # noqa: F401
def marginal(self, dimensions):
    """Return a marginal KDE distribution

        Parameters
        ----------
        dimensions : int or 1-d array_like
            The dimensions of the multivariate distribution corresponding
            with the marginal variables, that is, the indices of the dimensions
            that are being retained. The other dimensions are marginalized out.

        Returns
        -------
        marginal_kde : gaussian_kde
            An object representing the marginal distribution.

        Notes
        -----
        .. versionadded:: 1.10.0

        """
    dims = np.atleast_1d(dimensions)
    if not np.issubdtype(dims.dtype, np.integer):
        msg = 'Elements of `dimensions` must be integers - the indices of the marginal variables being retained.'
        raise ValueError(msg)
    n = len(self.dataset)
    original_dims = dims.copy()
    dims[dims < 0] = n + dims[dims < 0]
    if len(np.unique(dims)) != len(dims):
        msg = 'All elements of `dimensions` must be unique.'
        raise ValueError(msg)
    i_invalid = (dims < 0) | (dims >= n)
    if np.any(i_invalid):
        msg = f'Dimensions {original_dims[i_invalid]} are invalid for a distribution in {n} dimensions.'
        raise ValueError(msg)
    dataset = self.dataset[dims]
    weights = self.weights
    return gaussian_kde(dataset, bw_method=self.covariance_factor(), weights=weights)