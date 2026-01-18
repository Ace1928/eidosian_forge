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
def is_integer_slice(x):
    """
    Check that argument is an array of int.

    Parameters
    ----------
    x : object
        Object to check.

    Returns
    -------
    bool
        True if argument is an array of int, False otherwise.
    """
    if not is_slice(x):
        return False
    for pos in [x.start, x.stop, x.step]:
        if not (pos is None or is_integer(pos)):
            return False
    return True