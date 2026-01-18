from __future__ import annotations
import collections
from collections import abc
from collections.abc import (
import functools
from inspect import signature
from io import StringIO
import itertools
import operator
import sys
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from numpy import ma
from pandas._config import (
from pandas._config.config import _get_option
from pandas._libs import (
from pandas._libs.hashtable import duplicated
from pandas._libs.lib import is_range_indexer
from pandas.compat import PYPY
from pandas.compat._constants import REF_COUNT
from pandas.compat._optional import import_optional_dependency
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import (
from pandas.util._validators import (
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.accessor import CachedAccessor
from pandas.core.apply import reconstruct_and_relabel_result
from pandas.core.array_algos.take import take_2d_multi
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import (
from pandas.core.arrays.sparse import SparseFrameAccessor
from pandas.core.construction import (
from pandas.core.generic import (
from pandas.core.indexers import check_key_length
from pandas.core.indexes.api import (
from pandas.core.indexes.multi import (
from pandas.core.indexing import (
from pandas.core.internals import (
from pandas.core.internals.construction import (
from pandas.core.methods import selectn
from pandas.core.reshape.melt import melt
from pandas.core.series import Series
from pandas.core.shared_docs import _shared_docs
from pandas.core.sorting import (
from pandas.io.common import get_handle
from pandas.io.formats import (
from pandas.io.formats.info import (
import pandas.plotting
def set_index(self, keys, *, drop: bool=True, append: bool=False, inplace: bool=False, verify_integrity: bool=False) -> DataFrame | None:
    """
        Set the DataFrame index using existing columns.

        Set the DataFrame index (row labels) using one or more existing
        columns or arrays (of the correct length). The index can replace the
        existing index or expand on it.

        Parameters
        ----------
        keys : label or array-like or list of labels/arrays
            This parameter can be either a single column key, a single array of
            the same length as the calling DataFrame, or a list containing an
            arbitrary combination of column keys and arrays. Here, "array"
            encompasses :class:`Series`, :class:`Index`, ``np.ndarray``, and
            instances of :class:`~collections.abc.Iterator`.
        drop : bool, default True
            Delete columns to be used as the new index.
        append : bool, default False
            Whether to append columns to existing index.
        inplace : bool, default False
            Whether to modify the DataFrame rather than creating a new one.
        verify_integrity : bool, default False
            Check the new index for duplicates. Otherwise defer the check until
            necessary. Setting to False will improve the performance of this
            method.

        Returns
        -------
        DataFrame or None
            Changed row labels or None if ``inplace=True``.

        See Also
        --------
        DataFrame.reset_index : Opposite of set_index.
        DataFrame.reindex : Change to new indices or expand indices.
        DataFrame.reindex_like : Change to same indices as other DataFrame.

        Examples
        --------
        >>> df = pd.DataFrame({'month': [1, 4, 7, 10],
        ...                    'year': [2012, 2014, 2013, 2014],
        ...                    'sale': [55, 40, 84, 31]})
        >>> df
           month  year  sale
        0      1  2012    55
        1      4  2014    40
        2      7  2013    84
        3     10  2014    31

        Set the index to become the 'month' column:

        >>> df.set_index('month')
               year  sale
        month
        1      2012    55
        4      2014    40
        7      2013    84
        10     2014    31

        Create a MultiIndex using columns 'year' and 'month':

        >>> df.set_index(['year', 'month'])
                    sale
        year  month
        2012  1     55
        2014  4     40
        2013  7     84
        2014  10    31

        Create a MultiIndex using an Index and a column:

        >>> df.set_index([pd.Index([1, 2, 3, 4]), 'year'])
                 month  sale
           year
        1  2012  1      55
        2  2014  4      40
        3  2013  7      84
        4  2014  10     31

        Create a MultiIndex using two Series:

        >>> s = pd.Series([1, 2, 3, 4])
        >>> df.set_index([s, s**2])
              month  year  sale
        1 1       1  2012    55
        2 4       4  2014    40
        3 9       7  2013    84
        4 16     10  2014    31
        """
    inplace = validate_bool_kwarg(inplace, 'inplace')
    self._check_inplace_and_allows_duplicate_labels(inplace)
    if not isinstance(keys, list):
        keys = [keys]
    err_msg = 'The parameter "keys" may be a column key, one-dimensional array, or a list containing only valid column keys and one-dimensional arrays.'
    missing: list[Hashable] = []
    for col in keys:
        if isinstance(col, (Index, Series, np.ndarray, list, abc.Iterator)):
            if getattr(col, 'ndim', 1) != 1:
                raise ValueError(err_msg)
        else:
            try:
                found = col in self.columns
            except TypeError as err:
                raise TypeError(f'{err_msg}. Received column of type {type(col)}') from err
            else:
                if not found:
                    missing.append(col)
    if missing:
        raise KeyError(f'None of {missing} are in the columns')
    if inplace:
        frame = self
    else:
        frame = self.copy(deep=None)
    arrays: list[Index] = []
    names: list[Hashable] = []
    if append:
        names = list(self.index.names)
        if isinstance(self.index, MultiIndex):
            arrays.extend((self.index._get_level_values(i) for i in range(self.index.nlevels)))
        else:
            arrays.append(self.index)
    to_remove: list[Hashable] = []
    for col in keys:
        if isinstance(col, MultiIndex):
            arrays.extend((col._get_level_values(n) for n in range(col.nlevels)))
            names.extend(col.names)
        elif isinstance(col, (Index, Series)):
            arrays.append(col)
            names.append(col.name)
        elif isinstance(col, (list, np.ndarray)):
            arrays.append(col)
            names.append(None)
        elif isinstance(col, abc.Iterator):
            arrays.append(list(col))
            names.append(None)
        else:
            arrays.append(frame[col])
            names.append(col)
            if drop:
                to_remove.append(col)
        if len(arrays[-1]) != len(self):
            raise ValueError(f'Length mismatch: Expected {len(self)} rows, received array of length {len(arrays[-1])}')
    index = ensure_index_from_sequences(arrays, names)
    if verify_integrity and (not index.is_unique):
        duplicates = index[index.duplicated()].unique()
        raise ValueError(f'Index has duplicate keys: {duplicates}')
    for c in set(to_remove):
        del frame[c]
    index._cleanup()
    frame.index = index
    if not inplace:
        return frame
    return None