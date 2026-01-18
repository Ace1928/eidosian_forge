import numbers
import operator
import sys
import warnings
from contextlib import suppress
from functools import reduce, wraps
from inspect import Parameter, isclass, signature
import joblib
import numpy as np
import scipy.sparse as sp
from .. import get_config as _get_config
from ..exceptions import DataConversionWarning, NotFittedError, PositiveSpectrumWarning
from ..utils._array_api import _asarray_with_order, _is_numpy_namespace, get_namespace
from ..utils.fixes import ComplexWarning, _preserve_dia_indices_dtype
from ._isfinite import FiniteStatus, cy_isfinite
from .fixes import _object_dtype_isnan
def column_or_1d(y, *, dtype=None, warn=False):
    """Ravel column or 1d numpy array, else raises an error.

    Parameters
    ----------
    y : array-like
       Input data.

    dtype : data-type, default=None
        Data type for `y`.

        .. versionadded:: 1.2

    warn : bool, default=False
       To control display of warnings.

    Returns
    -------
    y : ndarray
       Output data.

    Raises
    ------
    ValueError
        If `y` is not a 1D array or a 2D array with a single row or column.

    Examples
    --------
    >>> from sklearn.utils.validation import column_or_1d
    >>> column_or_1d([1, 1])
    array([1, 1])
    """
    xp, _ = get_namespace(y)
    y = check_array(y, ensure_2d=False, dtype=dtype, input_name='y', force_all_finite=False, ensure_min_samples=0)
    shape = y.shape
    if len(shape) == 1:
        return _asarray_with_order(xp.reshape(y, (-1,)), order='C', xp=xp)
    if len(shape) == 2 and shape[1] == 1:
        if warn:
            warnings.warn('A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().', DataConversionWarning, stacklevel=2)
        return _asarray_with_order(xp.reshape(y, (-1,)), order='C', xp=xp)
    raise ValueError('y should be a 1d array, got an array of shape {} instead.'.format(shape))