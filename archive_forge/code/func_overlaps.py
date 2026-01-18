from __future__ import annotations
import operator
from operator import (
import textwrap
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.interval import (
from pandas._libs.missing import NA
from pandas._typing import (
from pandas.compat.numpy import function as nv
from pandas.errors import IntCastingNaNError
from pandas.util._decorators import Appender
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core.algorithms import (
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.arrays.base import (
from pandas.core.arrays.datetimes import DatetimeArray
from pandas.core.arrays.timedeltas import TimedeltaArray
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.indexers import check_array_indexer
from pandas.core.ops import (
from_arrays
from_tuples
from_breaks
@Appender(_interval_shared_docs['overlaps'] % {'klass': 'IntervalArray', 'examples': textwrap.dedent('        >>> data = [(0, 1), (1, 3), (2, 4)]\n        >>> intervals = pd.arrays.IntervalArray.from_tuples(data)\n        >>> intervals\n        <IntervalArray>\n        [(0, 1], (1, 3], (2, 4]]\n        Length: 3, dtype: interval[int64, right]\n        ')})
def overlaps(self, other):
    if isinstance(other, (IntervalArray, ABCIntervalIndex)):
        raise NotImplementedError
    if not isinstance(other, Interval):
        msg = f'`other` must be Interval-like, got {type(other).__name__}'
        raise TypeError(msg)
    op1 = le if self.closed_left and other.closed_right else lt
    op2 = le if other.closed_left and self.closed_right else lt
    return op1(self.left, other.right) & op2(other.left, self.right)