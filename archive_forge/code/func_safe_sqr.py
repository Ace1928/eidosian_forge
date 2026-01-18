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
def safe_sqr(X, *, copy=True):
    """Element wise squaring of array-likes and sparse matrices.

    Parameters
    ----------
    X : {array-like, ndarray, sparse matrix}

    copy : bool, default=True
        Whether to create a copy of X and operate on it or to perform
        inplace computation (default behaviour).

    Returns
    -------
    X ** 2 : element wise square
         Return the element-wise square of the input.

    Examples
    --------
    >>> from sklearn.utils import safe_sqr
    >>> safe_sqr([1, 2, 3])
    array([1, 4, 9])
    """
    X = check_array(X, accept_sparse=['csr', 'csc', 'coo'], ensure_2d=False)
    if issparse(X):
        if copy:
            X = X.copy()
        X.data **= 2
    elif copy:
        X = X ** 2
    else:
        X **= 2
    return X