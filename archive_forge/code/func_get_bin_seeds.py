import warnings
from collections import defaultdict
from numbers import Integral, Real
import numpy as np
from .._config import config_context
from ..base import BaseEstimator, ClusterMixin, _fit_context
from ..metrics.pairwise import pairwise_distances_argmin
from ..neighbors import NearestNeighbors
from ..utils import check_array, check_random_state, gen_batches
from ..utils._param_validation import Interval, validate_params
from ..utils.parallel import Parallel, delayed
from ..utils.validation import check_is_fitted
def get_bin_seeds(X, bin_size, min_bin_freq=1):
    """Find seeds for mean_shift.

    Finds seeds by first binning data onto a grid whose lines are
    spaced bin_size apart, and then choosing those bins with at least
    min_bin_freq points.

    Parameters
    ----------

    X : array-like of shape (n_samples, n_features)
        Input points, the same points that will be used in mean_shift.

    bin_size : float
        Controls the coarseness of the binning. Smaller values lead
        to more seeding (which is computationally more expensive). If you're
        not sure how to set this, set it to the value of the bandwidth used
        in clustering.mean_shift.

    min_bin_freq : int, default=1
        Only bins with at least min_bin_freq will be selected as seeds.
        Raising this value decreases the number of seeds found, which
        makes mean_shift computationally cheaper.

    Returns
    -------
    bin_seeds : array-like of shape (n_samples, n_features)
        Points used as initial kernel positions in clustering.mean_shift.
    """
    if bin_size == 0:
        return X
    bin_sizes = defaultdict(int)
    for point in X:
        binned_point = np.round(point / bin_size)
        bin_sizes[tuple(binned_point)] += 1
    bin_seeds = np.array([point for point, freq in bin_sizes.items() if freq >= min_bin_freq], dtype=np.float32)
    if len(bin_seeds) == len(X):
        warnings.warn('Binning data failed with provided bin_size=%f, using data points as seeds.' % bin_size)
        return X
    bin_seeds = bin_seeds * bin_size
    return bin_seeds