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
def pct_change(self, periods: int=1, fill_method: FillnaOptions | None | lib.NoDefault=lib.no_default, limit: int | None | lib.NoDefault=lib.no_default, freq=None, **kwargs) -> Self:
    """
        Fractional change between the current and a prior element.

        Computes the fractional change from the immediately previous row by
        default. This is useful in comparing the fraction of change in a time
        series of elements.

        .. note::

            Despite the name of this method, it calculates fractional change
            (also known as per unit change or relative change) and not
            percentage change. If you need the percentage change, multiply
            these values by 100.

        Parameters
        ----------
        periods : int, default 1
            Periods to shift for forming percent change.
        fill_method : {'backfill', 'bfill', 'pad', 'ffill', None}, default 'pad'
            How to handle NAs **before** computing percent changes.

            .. deprecated:: 2.1
                All options of `fill_method` are deprecated except `fill_method=None`.

        limit : int, default None
            The number of consecutive NAs to fill before stopping.

            .. deprecated:: 2.1

        freq : DateOffset, timedelta, or str, optional
            Increment to use from time series API (e.g. 'ME' or BDay()).
        **kwargs
            Additional keyword arguments are passed into
            `DataFrame.shift` or `Series.shift`.

        Returns
        -------
        Series or DataFrame
            The same type as the calling object.

        See Also
        --------
        Series.diff : Compute the difference of two elements in a Series.
        DataFrame.diff : Compute the difference of two elements in a DataFrame.
        Series.shift : Shift the index by some number of periods.
        DataFrame.shift : Shift the index by some number of periods.

        Examples
        --------
        **Series**

        >>> s = pd.Series([90, 91, 85])
        >>> s
        0    90
        1    91
        2    85
        dtype: int64

        >>> s.pct_change()
        0         NaN
        1    0.011111
        2   -0.065934
        dtype: float64

        >>> s.pct_change(periods=2)
        0         NaN
        1         NaN
        2   -0.055556
        dtype: float64

        See the percentage change in a Series where filling NAs with last
        valid observation forward to next valid.

        >>> s = pd.Series([90, 91, None, 85])
        >>> s
        0    90.0
        1    91.0
        2     NaN
        3    85.0
        dtype: float64

        >>> s.ffill().pct_change()
        0         NaN
        1    0.011111
        2    0.000000
        3   -0.065934
        dtype: float64

        **DataFrame**

        Percentage change in French franc, Deutsche Mark, and Italian lira from
        1980-01-01 to 1980-03-01.

        >>> df = pd.DataFrame({
        ...     'FR': [4.0405, 4.0963, 4.3149],
        ...     'GR': [1.7246, 1.7482, 1.8519],
        ...     'IT': [804.74, 810.01, 860.13]},
        ...     index=['1980-01-01', '1980-02-01', '1980-03-01'])
        >>> df
                        FR      GR      IT
        1980-01-01  4.0405  1.7246  804.74
        1980-02-01  4.0963  1.7482  810.01
        1980-03-01  4.3149  1.8519  860.13

        >>> df.pct_change()
                          FR        GR        IT
        1980-01-01       NaN       NaN       NaN
        1980-02-01  0.013810  0.013684  0.006549
        1980-03-01  0.053365  0.059318  0.061876

        Percentage of change in GOOG and APPL stock volume. Shows computing
        the percentage change between columns.

        >>> df = pd.DataFrame({
        ...     '2016': [1769950, 30586265],
        ...     '2015': [1500923, 40912316],
        ...     '2014': [1371819, 41403351]},
        ...     index=['GOOG', 'APPL'])
        >>> df
                  2016      2015      2014
        GOOG   1769950   1500923   1371819
        APPL  30586265  40912316  41403351

        >>> df.pct_change(axis='columns', periods=-1)
                  2016      2015  2014
        GOOG  0.179241  0.094112   NaN
        APPL -0.252395 -0.011860   NaN
        """
    if fill_method not in (lib.no_default, None) or limit is not lib.no_default:
        warnings.warn(f"The 'fill_method' keyword being not None and the 'limit' keyword in {type(self).__name__}.pct_change are deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.", FutureWarning, stacklevel=find_stack_level())
    if fill_method is lib.no_default:
        if limit is lib.no_default:
            cols = self.items() if self.ndim == 2 else [(None, self)]
            for _, col in cols:
                if len(col) > 0:
                    mask = col.isna().values
                    mask = mask[np.argmax(~mask):]
                    if mask.any():
                        warnings.warn(f"The default fill_method='pad' in {type(self).__name__}.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.", FutureWarning, stacklevel=find_stack_level())
                        break
        fill_method = 'pad'
    if limit is lib.no_default:
        limit = None
    axis = self._get_axis_number(kwargs.pop('axis', 'index'))
    if fill_method is None:
        data = self
    else:
        data = self._pad_or_backfill(fill_method, axis=axis, limit=limit)
    shifted = data.shift(periods=periods, freq=freq, axis=axis, **kwargs)
    rs = data / shifted - 1
    if freq is not None:
        rs = rs.loc[~rs.index.duplicated()]
        rs = rs.reindex_like(data)
    return rs.__finalize__(self, method='pct_change')