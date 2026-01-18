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
def select_dtypes(self, include=None, exclude=None) -> Self:
    """
        Return a subset of the DataFrame's columns based on the column dtypes.

        Parameters
        ----------
        include, exclude : scalar or list-like
            A selection of dtypes or strings to be included/excluded. At least
            one of these parameters must be supplied.

        Returns
        -------
        DataFrame
            The subset of the frame including the dtypes in ``include`` and
            excluding the dtypes in ``exclude``.

        Raises
        ------
        ValueError
            * If both of ``include`` and ``exclude`` are empty
            * If ``include`` and ``exclude`` have overlapping elements
            * If any kind of string dtype is passed in.

        See Also
        --------
        DataFrame.dtypes: Return Series with the data type of each column.

        Notes
        -----
        * To select all *numeric* types, use ``np.number`` or ``'number'``
        * To select strings you must use the ``object`` dtype, but note that
          this will return *all* object dtype columns
        * See the `numpy dtype hierarchy
          <https://numpy.org/doc/stable/reference/arrays.scalars.html>`__
        * To select datetimes, use ``np.datetime64``, ``'datetime'`` or
          ``'datetime64'``
        * To select timedeltas, use ``np.timedelta64``, ``'timedelta'`` or
          ``'timedelta64'``
        * To select Pandas categorical dtypes, use ``'category'``
        * To select Pandas datetimetz dtypes, use ``'datetimetz'``
          or ``'datetime64[ns, tz]'``

        Examples
        --------
        >>> df = pd.DataFrame({'a': [1, 2] * 3,
        ...                    'b': [True, False] * 3,
        ...                    'c': [1.0, 2.0] * 3})
        >>> df
                a      b  c
        0       1   True  1.0
        1       2  False  2.0
        2       1   True  1.0
        3       2  False  2.0
        4       1   True  1.0
        5       2  False  2.0

        >>> df.select_dtypes(include='bool')
           b
        0  True
        1  False
        2  True
        3  False
        4  True
        5  False

        >>> df.select_dtypes(include=['float64'])
           c
        0  1.0
        1  2.0
        2  1.0
        3  2.0
        4  1.0
        5  2.0

        >>> df.select_dtypes(exclude=['int64'])
               b    c
        0   True  1.0
        1  False  2.0
        2   True  1.0
        3  False  2.0
        4   True  1.0
        5  False  2.0
        """
    if not is_list_like(include):
        include = (include,) if include is not None else ()
    if not is_list_like(exclude):
        exclude = (exclude,) if exclude is not None else ()
    selection = (frozenset(include), frozenset(exclude))
    if not any(selection):
        raise ValueError('at least one of include or exclude must be nonempty')

    def check_int_infer_dtype(dtypes):
        converted_dtypes: list[type] = []
        for dtype in dtypes:
            if isinstance(dtype, str) and dtype == 'int' or dtype is int:
                converted_dtypes.append(np.int32)
                converted_dtypes.append(np.int64)
            elif dtype == 'float' or dtype is float:
                converted_dtypes.extend([np.float64, np.float32])
            else:
                converted_dtypes.append(infer_dtype_from_object(dtype))
        return frozenset(converted_dtypes)
    include = check_int_infer_dtype(include)
    exclude = check_int_infer_dtype(exclude)
    for dtypes in (include, exclude):
        invalidate_string_dtypes(dtypes)
    if not include.isdisjoint(exclude):
        raise ValueError(f'include and exclude overlap on {include & exclude}')

    def dtype_predicate(dtype: DtypeObj, dtypes_set) -> bool:
        dtype = dtype if not isinstance(dtype, ArrowDtype) else dtype.numpy_dtype
        return issubclass(dtype.type, tuple(dtypes_set)) or (np.number in dtypes_set and getattr(dtype, '_is_numeric', False) and (not is_bool_dtype(dtype)))

    def predicate(arr: ArrayLike) -> bool:
        dtype = arr.dtype
        if include:
            if not dtype_predicate(dtype, include):
                return False
        if exclude:
            if dtype_predicate(dtype, exclude):
                return False
        return True
    mgr = self._mgr._get_data_subset(predicate).copy(deep=None)
    return self._constructor_from_mgr(mgr, axes=mgr.axes).__finalize__(self)