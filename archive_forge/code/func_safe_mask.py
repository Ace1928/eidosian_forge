import math
import numbers
import platform
import struct
import timeit
import warnings
from collections.abc import Sequence
from contextlib import contextmanager, suppress
from itertools import compress, islice
import numpy as np
from scipy.sparse import issparse
from .. import get_config
from ..exceptions import DataConversionWarning
from . import _joblib, metadata_routing
from ._bunch import Bunch
from ._estimator_html_repr import estimator_html_repr
from ._param_validation import Integral, Interval, validate_params
from .class_weight import compute_class_weight, compute_sample_weight
from .deprecation import deprecated
from .discovery import all_estimators
from .fixes import parse_version, threadpool_info
from .murmurhash import murmurhash3_32
from .validation import (
@validate_params({'X': ['array-like', 'sparse matrix'], 'mask': ['array-like']}, prefer_skip_nested_validation=True)
def safe_mask(X, mask):
    """Return a mask which is safe to use on X.

    Parameters
    ----------
    X : {array-like, sparse matrix}
        Data on which to apply mask.

    mask : array-like
        Mask to be used on X.

    Returns
    -------
    mask : ndarray
        Array that is safe to use on X.

    Examples
    --------
    >>> from sklearn.utils import safe_mask
    >>> from scipy.sparse import csr_matrix
    >>> data = csr_matrix([[1], [2], [3], [4], [5]])
    >>> condition = [False, True, True, False, True]
    >>> mask = safe_mask(data, condition)
    >>> data[mask].toarray()
    array([[2],
           [3],
           [5]])
    """
    mask = np.asarray(mask)
    if np.issubdtype(mask.dtype, np.signedinteger):
        return mask
    if hasattr(X, 'toarray'):
        ind = np.arange(mask.shape[0])
        mask = ind[mask]
    return mask