import functools
from numbers import Integral
import numpy as np
from scipy.sparse import issparse
from ...preprocessing import LabelEncoder
from ...utils import _safe_indexing, check_random_state, check_X_y
from ...utils._param_validation import (
from ..pairwise import _VALID_METRICS, pairwise_distances, pairwise_distances_chunked
def check_number_of_labels(n_labels, n_samples):
    """Check that number of labels are valid.

    Parameters
    ----------
    n_labels : int
        Number of labels.

    n_samples : int
        Number of samples.
    """
    if not 1 < n_labels < n_samples:
        raise ValueError('Number of labels is %d. Valid values are 2 to n_samples - 1 (inclusive)' % n_labels)