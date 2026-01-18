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
def boolean_mask_to_numeric(indexer):
    """
    Convert boolean mask to numeric indices.

    Parameters
    ----------
    indexer : list-like of booleans

    Returns
    -------
    np.ndarray of ints
        Numerical positions of ``True`` elements in the passed `indexer`.
    """
    if isinstance(indexer, (np.ndarray, array, pandas.Series)):
        return np.where(indexer)[0]
    else:
        return np.fromiter(itertools.compress(data=range(len(indexer)), selectors=indexer), dtype=np.int64)