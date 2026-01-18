from __future__ import annotations
from collections.abc import (
import datetime
from functools import partial
from typing import (
import uuid
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.lib import is_range_indexer
from pandas._typing import (
from pandas.errors import MergeError
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas import (
import pandas.core.algorithms as algos
from pandas.core.arrays import (
from pandas.core.arrays.string_ import StringDtype
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.frame import _merge_doc
from pandas.core.indexes.api import default_index
from pandas.core.sorting import (
def get_join_indexers(left_keys: list[ArrayLike], right_keys: list[ArrayLike], sort: bool=False, how: JoinHow='inner') -> tuple[npt.NDArray[np.intp] | None, npt.NDArray[np.intp] | None]:
    """

    Parameters
    ----------
    left_keys : list[ndarray, ExtensionArray, Index, Series]
    right_keys : list[ndarray, ExtensionArray, Index, Series]
    sort : bool, default False
    how : {'inner', 'outer', 'left', 'right'}, default 'inner'

    Returns
    -------
    np.ndarray[np.intp] or None
        Indexer into the left_keys.
    np.ndarray[np.intp] or None
        Indexer into the right_keys.
    """
    assert len(left_keys) == len(right_keys), 'left_keys and right_keys must be the same length'
    left_n = len(left_keys[0])
    right_n = len(right_keys[0])
    if left_n == 0:
        if how in ['left', 'inner']:
            return _get_empty_indexer()
        elif not sort and how in ['right', 'outer']:
            return _get_no_sort_one_missing_indexer(right_n, True)
    elif right_n == 0:
        if how in ['right', 'inner']:
            return _get_empty_indexer()
        elif not sort and how in ['left', 'outer']:
            return _get_no_sort_one_missing_indexer(left_n, False)
    lkey: ArrayLike
    rkey: ArrayLike
    if len(left_keys) > 1:
        mapped = (_factorize_keys(left_keys[n], right_keys[n], sort=sort) for n in range(len(left_keys)))
        zipped = zip(*mapped)
        llab, rlab, shape = (list(x) for x in zipped)
        lkey, rkey = _get_join_keys(llab, rlab, tuple(shape), sort)
    else:
        lkey = left_keys[0]
        rkey = right_keys[0]
    left = Index(lkey)
    right = Index(rkey)
    if left.is_monotonic_increasing and right.is_monotonic_increasing and (left.is_unique or right.is_unique):
        _, lidx, ridx = left.join(right, how=how, return_indexers=True, sort=sort)
    else:
        lidx, ridx = get_join_indexers_non_unique(left._values, right._values, sort, how)
    if lidx is not None and is_range_indexer(lidx, len(left)):
        lidx = None
    if ridx is not None and is_range_indexer(ridx, len(right)):
        ridx = None
    return (lidx, ridx)