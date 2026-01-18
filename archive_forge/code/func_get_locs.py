from __future__ import annotations
from collections.abc import (
from functools import wraps
from sys import getsizeof
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import (
from pandas._libs.hashtable import duplicated
from pandas._typing import (
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import coerce_indexer_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import is_array_like
from pandas.core.dtypes.missing import (
import pandas.core.algorithms as algos
from pandas.core.array_algos.putmask import validate_putmask
from pandas.core.arrays import (
from pandas.core.arrays.categorical import (
import pandas.core.common as com
from pandas.core.construction import sanitize_array
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import (
from pandas.core.indexes.frozen import FrozenList
from pandas.core.ops.invalid import make_invalid_op
from pandas.core.sorting import (
from pandas.io.formats.printing import (
def get_locs(self, seq) -> npt.NDArray[np.intp]:
    """
        Get location for a sequence of labels.

        Parameters
        ----------
        seq : label, slice, list, mask or a sequence of such
           You should use one of the above for each level.
           If a level should not be used, set it to ``slice(None)``.

        Returns
        -------
        numpy.ndarray
            NumPy array of integers suitable for passing to iloc.

        See Also
        --------
        MultiIndex.get_loc : Get location for a label or a tuple of labels.
        MultiIndex.slice_locs : Get slice location given start label(s) and
                                end label(s).

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([list('abb'), list('def')])

        >>> mi.get_locs('b')  # doctest: +SKIP
        array([1, 2], dtype=int64)

        >>> mi.get_locs([slice(None), ['e', 'f']])  # doctest: +SKIP
        array([1, 2], dtype=int64)

        >>> mi.get_locs([[True, False, True], slice('e', 'f')])  # doctest: +SKIP
        array([2], dtype=int64)
        """
    true_slices = [i for i, s in enumerate(com.is_true_slices(seq)) if s]
    if true_slices and true_slices[-1] >= self._lexsort_depth:
        raise UnsortedIndexError(f'MultiIndex slicing requires the index to be lexsorted: slicing on levels {true_slices}, lexsort depth {self._lexsort_depth}')
    if any((x is Ellipsis for x in seq)):
        raise NotImplementedError('MultiIndex does not support indexing with Ellipsis')
    n = len(self)

    def _to_bool_indexer(indexer) -> npt.NDArray[np.bool_]:
        if isinstance(indexer, slice):
            new_indexer = np.zeros(n, dtype=np.bool_)
            new_indexer[indexer] = True
            return new_indexer
        return indexer
    indexer: npt.NDArray[np.bool_] | None = None
    for i, k in enumerate(seq):
        lvl_indexer: npt.NDArray[np.bool_] | slice | None = None
        if com.is_bool_indexer(k):
            if len(k) != n:
                raise ValueError('cannot index with a boolean indexer that is not the same length as the index')
            lvl_indexer = np.asarray(k)
            if indexer is None:
                lvl_indexer = lvl_indexer.copy()
        elif is_list_like(k):
            try:
                lvl_indexer = self._get_level_indexer(k, level=i, indexer=indexer)
            except (InvalidIndexError, TypeError, KeyError) as err:
                for x in k:
                    if not is_hashable(x):
                        raise err
                    item_indexer = self._get_level_indexer(x, level=i, indexer=indexer)
                    if lvl_indexer is None:
                        lvl_indexer = _to_bool_indexer(item_indexer)
                    elif isinstance(item_indexer, slice):
                        lvl_indexer[item_indexer] = True
                    else:
                        lvl_indexer |= item_indexer
            if lvl_indexer is None:
                return np.array([], dtype=np.intp)
        elif com.is_null_slice(k):
            if indexer is None and i == len(seq) - 1:
                return np.arange(n, dtype=np.intp)
            continue
        else:
            lvl_indexer = self._get_level_indexer(k, level=i, indexer=indexer)
        lvl_indexer = _to_bool_indexer(lvl_indexer)
        if indexer is None:
            indexer = lvl_indexer
        else:
            indexer &= lvl_indexer
            if not np.any(indexer) and np.any(lvl_indexer):
                raise KeyError(seq)
    if indexer is None:
        return np.array([], dtype=np.intp)
    pos_indexer = indexer.nonzero()[0]
    return self._reorder_indexer(seq, pos_indexer)