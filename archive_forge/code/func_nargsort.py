from __future__ import annotations
from collections import defaultdict
from typing import (
import numpy as np
from pandas._libs import (
from pandas._libs.hashtable import unique_label_indices
from pandas.core.dtypes.common import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import isna
from pandas.core.construction import extract_array
def nargsort(items: ArrayLike | Index | Series, kind: SortKind='quicksort', ascending: bool=True, na_position: str='last', key: Callable | None=None, mask: npt.NDArray[np.bool_] | None=None) -> npt.NDArray[np.intp]:
    """
    Intended to be a drop-in replacement for np.argsort which handles NaNs.

    Adds ascending, na_position, and key parameters.

    (GH #6399, #5231, #27237)

    Parameters
    ----------
    items : np.ndarray, ExtensionArray, Index, or Series
    kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, default 'quicksort'
    ascending : bool, default True
    na_position : {'first', 'last'}, default 'last'
    key : Optional[Callable], default None
    mask : Optional[np.ndarray[bool]], default None
        Passed when called by ExtensionArray.argsort.

    Returns
    -------
    np.ndarray[np.intp]
    """
    if key is not None:
        items = ensure_key_mapped(items, key)
        return nargsort(items, kind=kind, ascending=ascending, na_position=na_position, key=None, mask=mask)
    if isinstance(items, ABCRangeIndex):
        return items.argsort(ascending=ascending)
    elif not isinstance(items, ABCMultiIndex):
        items = extract_array(items)
    else:
        raise TypeError('nargsort does not support MultiIndex. Use index.sort_values instead.')
    if mask is None:
        mask = np.asarray(isna(items))
    if not isinstance(items, np.ndarray):
        return items.argsort(ascending=ascending, kind=kind, na_position=na_position)
    idx = np.arange(len(items))
    non_nans = items[~mask]
    non_nan_idx = idx[~mask]
    nan_idx = np.nonzero(mask)[0]
    if not ascending:
        non_nans = non_nans[::-1]
        non_nan_idx = non_nan_idx[::-1]
    indexer = non_nan_idx[non_nans.argsort(kind=kind)]
    if not ascending:
        indexer = indexer[::-1]
    if na_position == 'last':
        indexer = np.concatenate([indexer, nan_idx])
    elif na_position == 'first':
        indexer = np.concatenate([nan_idx, indexer])
    else:
        raise ValueError(f'invalid na_position: {na_position}')
    return ensure_platform_int(indexer)