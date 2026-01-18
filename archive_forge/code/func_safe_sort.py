from __future__ import annotations
import decimal
import operator
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._typing import (
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core.array_algos.take import take_nd
from pandas.core.construction import (
from pandas.core.indexers import validate_indices
def safe_sort(values: Index | ArrayLike, codes: npt.NDArray[np.intp] | None=None, use_na_sentinel: bool=True, assume_unique: bool=False, verify: bool=True) -> AnyArrayLike | tuple[AnyArrayLike, np.ndarray]:
    """
    Sort ``values`` and reorder corresponding ``codes``.

    ``values`` should be unique if ``codes`` is not None.
    Safe for use with mixed types (int, str), orders ints before strs.

    Parameters
    ----------
    values : list-like
        Sequence; must be unique if ``codes`` is not None.
    codes : np.ndarray[intp] or None, default None
        Indices to ``values``. All out of bound indices are treated as
        "not found" and will be masked with ``-1``.
    use_na_sentinel : bool, default True
        If True, the sentinel -1 will be used for NaN values. If False,
        NaN values will be encoded as non-negative integers and will not drop the
        NaN from the uniques of the values.
    assume_unique : bool, default False
        When True, ``values`` are assumed to be unique, which can speed up
        the calculation. Ignored when ``codes`` is None.
    verify : bool, default True
        Check if codes are out of bound for the values and put out of bound
        codes equal to ``-1``. If ``verify=False``, it is assumed there
        are no out of bound codes. Ignored when ``codes`` is None.

    Returns
    -------
    ordered : AnyArrayLike
        Sorted ``values``
    new_codes : ndarray
        Reordered ``codes``; returned when ``codes`` is not None.

    Raises
    ------
    TypeError
        * If ``values`` is not list-like or if ``codes`` is neither None
        nor list-like
        * If ``values`` cannot be sorted
    ValueError
        * If ``codes`` is not None and ``values`` contain duplicates.
    """
    if not isinstance(values, (np.ndarray, ABCExtensionArray, ABCIndex)):
        raise TypeError('Only np.ndarray, ExtensionArray, and Index objects are allowed to be passed to safe_sort as values')
    sorter = None
    ordered: AnyArrayLike
    if not isinstance(values.dtype, ExtensionDtype) and lib.infer_dtype(values, skipna=False) == 'mixed-integer':
        ordered = _sort_mixed(values)
    else:
        try:
            sorter = values.argsort()
            ordered = values.take(sorter)
        except (TypeError, decimal.InvalidOperation):
            if values.size and isinstance(values[0], tuple):
                ordered = _sort_tuples(values)
            else:
                ordered = _sort_mixed(values)
    if codes is None:
        return ordered
    if not is_list_like(codes):
        raise TypeError('Only list-like objects or None are allowed to be passed to safe_sort as codes')
    codes = ensure_platform_int(np.asarray(codes))
    if not assume_unique and (not len(unique(values)) == len(values)):
        raise ValueError('values should be unique if codes is not None')
    if sorter is None:
        hash_klass, values = _get_hashtable_algo(values)
        t = hash_klass(len(values))
        t.map_locations(values)
        sorter = ensure_platform_int(t.lookup(ordered))
    if use_na_sentinel:
        order2 = sorter.argsort()
        if verify:
            mask = (codes < -len(values)) | (codes >= len(values))
            codes[mask] = 0
        else:
            mask = None
        new_codes = take_nd(order2, codes, fill_value=-1)
    else:
        reverse_indexer = np.empty(len(sorter), dtype=int)
        reverse_indexer.put(sorter, np.arange(len(sorter)))
        new_codes = reverse_indexer.take(codes, mode='wrap')
        if use_na_sentinel:
            mask = codes == -1
            if verify:
                mask = mask | (codes < -len(values)) | (codes >= len(values))
    if use_na_sentinel and mask is not None:
        np.putmask(new_codes, mask, -1)
    return (ordered, ensure_platform_int(new_codes))