from __future__ import annotations
from typing import TYPE_CHECKING
from pandas._libs import lib
from pandas.core.dtypes.missing import notna
from pandas.core.algorithms import factorize
from pandas.core.indexes.api import MultiIndex
from pandas.core.series import Series
def coo_to_sparse_series(A: scipy.sparse.coo_matrix, dense_index: bool=False) -> Series:
    """
    Convert a scipy.sparse.coo_matrix to a Series with type sparse.

    Parameters
    ----------
    A : scipy.sparse.coo_matrix
    dense_index : bool, default False

    Returns
    -------
    Series

    Raises
    ------
    TypeError if A is not a coo_matrix
    """
    from pandas import SparseDtype
    try:
        ser = Series(A.data, MultiIndex.from_arrays((A.row, A.col)), copy=False)
    except AttributeError as err:
        raise TypeError(f'Expected coo_matrix. Got {type(A).__name__} instead.') from err
    ser = ser.sort_index()
    ser = ser.astype(SparseDtype(ser.dtype))
    if dense_index:
        ind = MultiIndex.from_product([A.row, A.col])
        ser = ser.reindex(ind)
    return ser