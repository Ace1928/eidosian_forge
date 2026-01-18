import os.path
from contextlib import closing
from numbers import Integral
import numpy as np
import scipy.sparse as sp
from .. import __version__
from ..utils import IS_PYPY, check_array
from ..utils._param_validation import HasMethods, Interval, StrOptions, validate_params
@validate_params({'files': ['array-like', str, os.PathLike, HasMethods('read'), Interval(Integral, 0, None, closed='left')], 'n_features': [Interval(Integral, 1, None, closed='left'), None], 'dtype': 'no_validation', 'multilabel': ['boolean'], 'zero_based': ['boolean', StrOptions({'auto'})], 'query_id': ['boolean'], 'offset': [Interval(Integral, 0, None, closed='left')], 'length': [Integral]}, prefer_skip_nested_validation=True)
def load_svmlight_files(files, *, n_features=None, dtype=np.float64, multilabel=False, zero_based='auto', query_id=False, offset=0, length=-1):
    """Load dataset from multiple files in SVMlight format.

    This function is equivalent to mapping load_svmlight_file over a list of
    files, except that the results are concatenated into a single, flat list
    and the samples vectors are constrained to all have the same number of
    features.

    In case the file contains a pairwise preference constraint (known
    as "qid" in the svmlight format) these are ignored unless the
    query_id parameter is set to True. These pairwise preference
    constraints can be used to constraint the combination of samples
    when using pairwise loss functions (as is the case in some
    learning to rank problems) so that only pairs with the same
    query_id value are considered.

    Parameters
    ----------
    files : array-like, dtype=str, path-like, file-like or int
        (Paths of) files to load. If a path ends in ".gz" or ".bz2", it will
        be uncompressed on the fly. If an integer is passed, it is assumed to
        be a file descriptor. File-likes and file descriptors will not be
        closed by this function. File-like objects must be opened in binary
        mode.

        .. versionchanged:: 1.2
           Path-like objects are now accepted.

    n_features : int, default=None
        The number of features to use. If None, it will be inferred from the
        maximum column index occurring in any of the files.

        This can be set to a higher value than the actual number of features
        in any of the input files, but setting it to a lower value will cause
        an exception to be raised.

    dtype : numpy data type, default=np.float64
        Data type of dataset to be loaded. This will be the data type of the
        output numpy arrays ``X`` and ``y``.

    multilabel : bool, default=False
        Samples may have several labels each (see
        https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html).

    zero_based : bool or "auto", default="auto"
        Whether column indices in f are zero-based (True) or one-based
        (False). If column indices are one-based, they are transformed to
        zero-based to match Python/NumPy conventions.
        If set to "auto", a heuristic check is applied to determine this from
        the file contents. Both kinds of files occur "in the wild", but they
        are unfortunately not self-identifying. Using "auto" or True should
        always be safe when no offset or length is passed.
        If offset or length are passed, the "auto" mode falls back
        to zero_based=True to avoid having the heuristic check yield
        inconsistent results on different segments of the file.

    query_id : bool, default=False
        If True, will return the query_id array for each file.

    offset : int, default=0
        Ignore the offset first bytes by seeking forward, then
        discarding the following bytes up until the next new line
        character.

    length : int, default=-1
        If strictly positive, stop reading any new line of data once the
        position in the file has reached the (offset + length) bytes threshold.

    Returns
    -------
    [X1, y1, ..., Xn, yn] or [X1, y1, q1, ..., Xn, yn, qn]: list of arrays
        Each (Xi, yi) pair is the result from load_svmlight_file(files[i]).
        If query_id is set to True, this will return instead (Xi, yi, qi)
        triplets.

    See Also
    --------
    load_svmlight_file: Similar function for loading a single file in this
        format.

    Notes
    -----
    When fitting a model to a matrix X_train and evaluating it against a
    matrix X_test, it is essential that X_train and X_test have the same
    number of features (X_train.shape[1] == X_test.shape[1]). This may not
    be the case if you load the files individually with load_svmlight_file.
    """
    if (offset != 0 or length > 0) and zero_based == 'auto':
        zero_based = True
    if (offset != 0 or length > 0) and n_features is None:
        raise ValueError('n_features is required when offset or length is specified.')
    r = [_open_and_load(f, dtype, multilabel, bool(zero_based), bool(query_id), offset=offset, length=length) for f in files]
    if zero_based is False or (zero_based == 'auto' and all((len(tmp[1]) and np.min(tmp[1]) > 0 for tmp in r))):
        for _, indices, _, _, _ in r:
            indices -= 1
    n_f = max((ind[1].max() if len(ind[1]) else 0 for ind in r)) + 1
    if n_features is None:
        n_features = n_f
    elif n_features < n_f:
        raise ValueError('n_features was set to {}, but input file contains {} features'.format(n_features, n_f))
    result = []
    for data, indices, indptr, y, query_values in r:
        shape = (indptr.shape[0] - 1, n_features)
        X = sp.csr_matrix((data, indices, indptr), shape)
        X.sort_indices()
        result += (X, y)
        if query_id:
            result.append(query_values)
    return result