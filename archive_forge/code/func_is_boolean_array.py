import itertools
import numpy as np
import pandas
from pandas.api.types import is_bool, is_list_like
from pandas.core.dtypes.common import is_bool_dtype, is_integer, is_integer_dtype
from pandas.core.indexing import IndexingError
from modin.error_message import ErrorMessage
from modin.pandas.indexing import compute_sliced_len, is_range_like, is_slice, is_tuple
from modin.pandas.utils import is_scalar
from .arr import array
def is_boolean_array(x):
    """
    Check that argument is an array of bool.

    Parameters
    ----------
    x : object
        Object to check.

    Returns
    -------
    bool
        True if argument is an array of bool, False otherwise.
    """
    if isinstance(x, (np.ndarray, array, pandas.Series, pandas.Index)):
        return is_bool_dtype(x.dtype)
    return is_list_like(x) and all(map(is_bool, x))