from __future__ import annotations
from operator import (
import textwrap
from typing import (
import numpy as np
from pandas._libs import lib
from pandas._libs.interval import (
from pandas._libs.tslibs import (
from pandas.errors import InvalidIndexError
from pandas.util._decorators import (
from pandas.util._exceptions import rewrite_exception
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import is_valid_na_for_dtype
from pandas.core.algorithms import unique
from pandas.core.arrays.datetimelike import validate_periods
from pandas.core.arrays.interval import (
import pandas.core.common as com
from pandas.core.indexers import is_valid_positional_slice
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import (
from pandas.core.indexes.datetimes import (
from pandas.core.indexes.extension import (
from pandas.core.indexes.multi import MultiIndex
from pandas.core.indexes.timedeltas import (
@property
def is_overlapping(self) -> bool:
    """
        Return True if the IntervalIndex has overlapping intervals, else False.

        Two intervals overlap if they share a common point, including closed
        endpoints. Intervals that only have an open endpoint in common do not
        overlap.

        Returns
        -------
        bool
            Boolean indicating if the IntervalIndex has overlapping intervals.

        See Also
        --------
        Interval.overlaps : Check whether two Interval objects overlap.
        IntervalIndex.overlaps : Check an IntervalIndex elementwise for
            overlaps.

        Examples
        --------
        >>> index = pd.IntervalIndex.from_tuples([(0, 2), (1, 3), (4, 5)])
        >>> index
        IntervalIndex([(0, 2], (1, 3], (4, 5]],
              dtype='interval[int64, right]')
        >>> index.is_overlapping
        True

        Intervals that share closed endpoints overlap:

        >>> index = pd.interval_range(0, 3, closed='both')
        >>> index
        IntervalIndex([[0, 1], [1, 2], [2, 3]],
              dtype='interval[int64, both]')
        >>> index.is_overlapping
        True

        Intervals that only have an open endpoint in common do not overlap:

        >>> index = pd.interval_range(0, 3, closed='left')
        >>> index
        IntervalIndex([[0, 1), [1, 2), [2, 3)],
              dtype='interval[int64, left]')
        >>> index.is_overlapping
        False
        """
    return self._engine.is_overlapping