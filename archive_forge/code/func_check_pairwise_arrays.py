import itertools
import warnings
from functools import partial
from numbers import Integral, Real
import numpy as np
from joblib import effective_n_jobs
from scipy.sparse import csr_matrix, issparse
from scipy.spatial import distance
from .. import config_context
from ..exceptions import DataConversionWarning
from ..preprocessing import normalize
from ..utils import (
from ..utils._mask import _get_mask
from ..utils._param_validation import (
from ..utils.extmath import row_norms, safe_sparse_dot
from ..utils.fixes import parse_version, sp_base_version
from ..utils.parallel import Parallel, delayed
from ..utils.validation import _num_samples, check_non_negative
from ._pairwise_distances_reduction import ArgKmin
from ._pairwise_fast import _chi2_kernel_fast, _sparse_manhattan
def check_pairwise_arrays(X, Y, *, precomputed=False, dtype=None, accept_sparse='csr', force_all_finite=True, copy=False):
    """Set X and Y appropriately and checks inputs.

    If Y is None, it is set as a pointer to X (i.e. not a copy).
    If Y is given, this does not happen.
    All distance metrics should use this function first to assert that the
    given parameters are correct and safe to use.

    Specifically, this function first ensures that both X and Y are arrays,
    then checks that they are at least two dimensional while ensuring that
    their elements are floats (or dtype if provided). Finally, the function
    checks that the size of the second dimension of the two arrays is equal, or
    the equivalent check for a precomputed distance matrix.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_X, n_features)

    Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features)

    precomputed : bool, default=False
        True if X is to be treated as precomputed distances to the samples in
        Y.

    dtype : str, type, list of type, default=None
        Data type required for X and Y. If None, the dtype will be an
        appropriate float type selected by _return_float_dtype.

        .. versionadded:: 0.18

    accept_sparse : str, bool or list/tuple of str, default='csr'
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.

    force_all_finite : bool or 'allow-nan', default=True
        Whether to raise an error on np.inf, np.nan, pd.NA in array. The
        possibilities are:

        - True: Force all values of array to be finite.
        - False: accepts np.inf, np.nan, pd.NA in array.
        - 'allow-nan': accepts only np.nan and pd.NA values in array. Values
          cannot be infinite.

        .. versionadded:: 0.22
           ``force_all_finite`` accepts the string ``'allow-nan'``.

        .. versionchanged:: 0.23
           Accepts `pd.NA` and converts it into `np.nan`.

    copy : bool, default=False
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

        .. versionadded:: 0.22

    Returns
    -------
    safe_X : {array-like, sparse matrix} of shape (n_samples_X, n_features)
        An array equal to X, guaranteed to be a numpy array.

    safe_Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features)
        An array equal to Y if Y was not None, guaranteed to be a numpy array.
        If Y was None, safe_Y will be a pointer to X.
    """
    X, Y, dtype_float = _return_float_dtype(X, Y)
    estimator = 'check_pairwise_arrays'
    if dtype is None:
        dtype = dtype_float
    if Y is X or Y is None:
        X = Y = check_array(X, accept_sparse=accept_sparse, dtype=dtype, copy=copy, force_all_finite=force_all_finite, estimator=estimator)
    else:
        X = check_array(X, accept_sparse=accept_sparse, dtype=dtype, copy=copy, force_all_finite=force_all_finite, estimator=estimator)
        Y = check_array(Y, accept_sparse=accept_sparse, dtype=dtype, copy=copy, force_all_finite=force_all_finite, estimator=estimator)
    if precomputed:
        if X.shape[1] != Y.shape[0]:
            raise ValueError('Precomputed metric requires shape (n_queries, n_indexed). Got (%d, %d) for %d indexed.' % (X.shape[0], X.shape[1], Y.shape[0]))
    elif X.shape[1] != Y.shape[1]:
        raise ValueError('Incompatible dimension for X and Y matrices: X.shape[1] == %d while Y.shape[1] == %d' % (X.shape[1], Y.shape[1]))
    return (X, Y)