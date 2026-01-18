from __future__ import annotations
import collections
from copy import deepcopy
import datetime as dt
from functools import partial
import gc
from json import loads
import operator
import pickle
import re
import sys
from typing import (
import warnings
import weakref
import numpy as np
from pandas._config import (
from pandas._libs import lib
from pandas._libs.lib import is_range_indexer
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
from pandas._typing import (
from pandas.compat import PYPY
from pandas.compat._constants import REF_COUNT
from pandas.compat._optional import import_optional_dependency
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import (
from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.array_algos.replace import should_use_regex
from pandas.core.arrays import ExtensionArray
from pandas.core.base import PandasObject
from pandas.core.construction import extract_array
from pandas.core.flags import Flags
from pandas.core.indexes.api import (
from pandas.core.internals import (
from pandas.core.internals.construction import (
from pandas.core.methods.describe import describe_ndframe
from pandas.core.missing import (
from pandas.core.reshape.concat import concat
from pandas.core.shared_docs import _shared_docs
from pandas.core.sorting import get_indexer_indexer
from pandas.core.window import (
from pandas.io.formats.format import (
from pandas.io.formats.printing import pprint_thing
@final
@doc(klass=_shared_doc_kwargs['klass'], axes_single_arg=_shared_doc_kwargs['axes_single_arg'])
def bfill(self, *, axis: None | Axis=None, inplace: bool_t=False, limit: None | int=None, limit_area: Literal['inside', 'outside'] | None=None, downcast: dict | None | lib.NoDefault=lib.no_default) -> Self | None:
    """
        Fill NA/NaN values by using the next valid observation to fill the gap.

        Parameters
        ----------
        axis : {axes_single_arg}
            Axis along which to fill missing values. For `Series`
            this parameter is unused and defaults to 0.
        inplace : bool, default False
            If True, fill in-place. Note: this will modify any
            other views on this object (e.g., a no-copy slice for a column in a
            DataFrame).
        limit : int, default None
            If method is specified, this is the maximum number of consecutive
            NaN values to forward/backward fill. In other words, if there is
            a gap with more than this number of consecutive NaNs, it will only
            be partially filled. If method is not specified, this is the
            maximum number of entries along the entire axis where NaNs will be
            filled. Must be greater than 0 if not None.
        limit_area : {{`None`, 'inside', 'outside'}}, default None
            If limit is specified, consecutive NaNs will be filled with this
            restriction.

            * ``None``: No fill restriction.
            * 'inside': Only fill NaNs surrounded by valid values
              (interpolate).
            * 'outside': Only fill NaNs outside valid values (extrapolate).

            .. versionadded:: 2.2.0

        downcast : dict, default is None
            A dict of item->dtype of what to downcast if possible,
            or the string 'infer' which will try to downcast to an appropriate
            equal type (e.g. float64 to int64 if possible).

            .. deprecated:: 2.2.0

        Returns
        -------
        {klass} or None
            Object with missing values filled or None if ``inplace=True``.

        Examples
        --------
        For Series:

        >>> s = pd.Series([1, None, None, 2])
        >>> s.bfill()
        0    1.0
        1    2.0
        2    2.0
        3    2.0
        dtype: float64
        >>> s.bfill(limit=1)
        0    1.0
        1    NaN
        2    2.0
        3    2.0
        dtype: float64

        With DataFrame:

        >>> df = pd.DataFrame({{'A': [1, None, None, 4], 'B': [None, 5, None, 7]}})
        >>> df
              A     B
        0   1.0	  NaN
        1   NaN	  5.0
        2   NaN   NaN
        3   4.0   7.0
        >>> df.bfill()
              A     B
        0   1.0   5.0
        1   4.0   5.0
        2   4.0   7.0
        3   4.0   7.0
        >>> df.bfill(limit=1)
              A     B
        0   1.0   5.0
        1   NaN   5.0
        2   4.0   7.0
        3   4.0   7.0
        """
    downcast = self._deprecate_downcast(downcast, 'bfill')
    inplace = validate_bool_kwarg(inplace, 'inplace')
    if inplace:
        if not PYPY and using_copy_on_write():
            if sys.getrefcount(self) <= REF_COUNT:
                warnings.warn(_chained_assignment_method_msg, ChainedAssignmentError, stacklevel=2)
        elif not PYPY and (not using_copy_on_write()) and self._is_view_after_cow_rules():
            ctr = sys.getrefcount(self)
            ref_count = REF_COUNT
            if isinstance(self, ABCSeries) and _check_cacher(self):
                ref_count += 1
            if ctr <= ref_count:
                warnings.warn(_chained_assignment_warning_method_msg, FutureWarning, stacklevel=2)
    return self._pad_or_backfill('bfill', axis=axis, inplace=inplace, limit=limit, limit_area=limit_area, downcast=downcast)