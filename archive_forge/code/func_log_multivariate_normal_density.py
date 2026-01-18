from __future__ import absolute_import, division, print_function
import numpy as np
from scipy import linalg
def log_multivariate_normal_density(x, means, covars, covariance_type='diag'):
    """
    Compute the log probability under a multivariate Gaussian distribution.

    Parameters
    ----------
    x : array_like, shape (n_samples, n_features)
        List of n_features-dimensional data points.  Each row corresponds to a
        single data point.
    means : array_like, shape (n_components, n_features)
        List of n_features-dimensional mean vectors for n_components Gaussians.
        Each row corresponds to a single mean vector.
    covars : array_like
        List of n_components covariance parameters for each Gaussian. The shape
        depends on `covariance_type`:

        - (n_components, n_features)             if 'spherical',
        - (n_features, n_features)               if 'tied',
        - (n_components, n_features)             if 'diag',
        - (n_components, n_features, n_features) if 'full'.

    covariance_type : {'diag', 'spherical', 'tied', 'full'}
        Type of the covariance parameters. Defaults to 'diag'.

    Returns
    -------
    lpr : array_like, shape (n_samples, n_components)
        Array containing the log probabilities of each data point in `x`
        under each of the n_components multivariate Gaussian distributions.

    """
    log_multivariate_normal_density_dict = {'spherical': _log_multivariate_normal_density_spherical, 'tied': _log_multivariate_normal_density_tied, 'diag': _log_multivariate_normal_density_diag, 'full': _log_multivariate_normal_density_full}
    return log_multivariate_normal_density_dict[covariance_type](x, means, covars)