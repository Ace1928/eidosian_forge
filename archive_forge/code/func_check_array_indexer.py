from __future__ import annotations
from typing import (
import numpy as np
from pandas._libs import lib
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
def check_array_indexer(array: AnyArrayLike, indexer: Any) -> Any:
    """
    Check if `indexer` is a valid array indexer for `array`.

    For a boolean mask, `array` and `indexer` are checked to have the same
    length. The dtype is validated, and if it is an integer or boolean
    ExtensionArray, it is checked if there are missing values present, and
    it is converted to the appropriate numpy array. Other dtypes will raise
    an error.

    Non-array indexers (integer, slice, Ellipsis, tuples, ..) are passed
    through as is.

    Parameters
    ----------
    array : array-like
        The array that is being indexed (only used for the length).
    indexer : array-like or list-like
        The array-like that's used to index. List-like input that is not yet
        a numpy array or an ExtensionArray is converted to one. Other input
        types are passed through as is.

    Returns
    -------
    numpy.ndarray
        The validated indexer as a numpy array that can be used to index.

    Raises
    ------
    IndexError
        When the lengths don't match.
    ValueError
        When `indexer` cannot be converted to a numpy ndarray to index
        (e.g. presence of missing values).

    See Also
    --------
    api.types.is_bool_dtype : Check if `key` is of boolean dtype.

    Examples
    --------
    When checking a boolean mask, a boolean ndarray is returned when the
    arguments are all valid.

    >>> mask = pd.array([True, False])
    >>> arr = pd.array([1, 2])
    >>> pd.api.indexers.check_array_indexer(arr, mask)
    array([ True, False])

    An IndexError is raised when the lengths don't match.

    >>> mask = pd.array([True, False, True])
    >>> pd.api.indexers.check_array_indexer(arr, mask)
    Traceback (most recent call last):
    ...
    IndexError: Boolean index has wrong length: 3 instead of 2.

    NA values in a boolean array are treated as False.

    >>> mask = pd.array([True, pd.NA])
    >>> pd.api.indexers.check_array_indexer(arr, mask)
    array([ True, False])

    A numpy boolean mask will get passed through (if the length is correct):

    >>> mask = np.array([True, False])
    >>> pd.api.indexers.check_array_indexer(arr, mask)
    array([ True, False])

    Similarly for integer indexers, an integer ndarray is returned when it is
    a valid indexer, otherwise an error is  (for integer indexers, a matching
    length is not required):

    >>> indexer = pd.array([0, 2], dtype="Int64")
    >>> arr = pd.array([1, 2, 3])
    >>> pd.api.indexers.check_array_indexer(arr, indexer)
    array([0, 2])

    >>> indexer = pd.array([0, pd.NA], dtype="Int64")
    >>> pd.api.indexers.check_array_indexer(arr, indexer)
    Traceback (most recent call last):
    ...
    ValueError: Cannot index with an integer indexer containing NA values

    For non-integer/boolean dtypes, an appropriate error is raised:

    >>> indexer = np.array([0., 2.], dtype="float64")
    >>> pd.api.indexers.check_array_indexer(arr, indexer)
    Traceback (most recent call last):
    ...
    IndexError: arrays used as indices must be of integer or boolean type
    """
    from pandas.core.construction import array as pd_array
    if is_list_like(indexer):
        if isinstance(indexer, tuple):
            return indexer
    else:
        return indexer
    if not is_array_like(indexer):
        indexer = pd_array(indexer)
        if len(indexer) == 0:
            indexer = np.array([], dtype=np.intp)
    dtype = indexer.dtype
    if is_bool_dtype(dtype):
        if isinstance(dtype, ExtensionDtype):
            indexer = indexer.to_numpy(dtype=bool, na_value=False)
        else:
            indexer = np.asarray(indexer, dtype=bool)
        if len(indexer) != len(array):
            raise IndexError(f'Boolean index has wrong length: {len(indexer)} instead of {len(array)}')
    elif is_integer_dtype(dtype):
        try:
            indexer = np.asarray(indexer, dtype=np.intp)
        except ValueError as err:
            raise ValueError('Cannot index with an integer indexer containing NA values') from err
    else:
        raise IndexError('arrays used as indices must be of integer or boolean type')
    return indexer