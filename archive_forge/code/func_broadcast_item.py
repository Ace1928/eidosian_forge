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
def broadcast_item(obj, row_lookup, col_lookup, item, need_columns_reindex=True):
    """
    Use NumPy to broadcast or reshape item with reindexing.

    Parameters
    ----------
    obj : DataFrame or Series
        The object containing the necessary information about the axes.
    row_lookup : slice or scalar
        The global row index to locate inside of `item`.
    col_lookup : range, array, list, slice or scalar
        The global col index to locate inside of `item`.
    item : DataFrame, Series, or query_compiler
        Value that should be broadcast to a new shape of `to_shape`.
    need_columns_reindex : bool, default: True
        In the case of assigning columns to a dataframe (broadcasting is
        part of the flow), reindexing is not needed.

    Returns
    -------
    np.ndarray
        `item` after it was broadcasted to `to_shape`.

    Raises
    ------
    ValueError
        1) If `row_lookup` or `col_lookup` contains values missing in
        DataFrame/Series index or columns correspondingly.
        2) If `item` cannot be broadcast from its own shape to `to_shape`.

    Notes
    -----
    NumPy is memory efficient, there shouldn't be performance issue.
    """
    new_row_len = len(obj._query_compiler.index[row_lookup]) if isinstance(row_lookup, slice) else len(row_lookup)
    new_col_len = len(obj._query_compiler.columns[col_lookup]) if isinstance(col_lookup, slice) else len(col_lookup)
    to_shape = (new_row_len, new_col_len)
    if isinstance(item, array):
        axes_to_reindex = {}
        index_values = obj._query_compiler.index[row_lookup]
        if not index_values.equals(item._query_compiler.index):
            axes_to_reindex['index'] = index_values
        if need_columns_reindex and isinstance(item, array) and (item._ndim == 2):
            column_values = obj._query_compiler.columns[col_lookup]
            if not column_values.equals(item._query_compiler.columns):
                axes_to_reindex['columns'] = column_values
        if axes_to_reindex:
            row_axes = axes_to_reindex.get('index', None)
            if row_axes is not None:
                item._query_compiler = item._query_compiler.reindex(axis=0, labels=row_axes, copy=None)
            col_axes = axes_to_reindex.get('columns', None)
            if col_axes is not None:
                item._query_compiler = item._query_compiler.reindex(axis=1, labels=col_axes, copy=None)
    try:
        item = np.array(item) if not isinstance(item, array) else item._to_numpy()
        if np.prod(to_shape) == np.prod(item.shape):
            return item.reshape(to_shape)
        else:
            return np.broadcast_to(item, to_shape)
    except ValueError:
        from_shape = np.array(item).shape
        raise ValueError(f'could not broadcast input array from shape {from_shape} into shape ' + f'{to_shape}')