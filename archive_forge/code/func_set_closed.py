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
@Appender(_interval_shared_docs['set_closed'] % {'klass': 'IntervalArray', 'examples': textwrap.dedent("        Examples\n        --------\n        >>> index = pd.arrays.IntervalArray.from_breaks(range(4))\n        >>> index\n        <IntervalArray>\n        [(0, 1], (1, 2], (2, 3]]\n        Length: 3, dtype: interval[int64, right]\n        >>> index.set_closed('both')\n        <IntervalArray>\n        [[0, 1], [1, 2], [2, 3]]\n        Length: 3, dtype: interval[int64, both]\n        ")})
def set_closed(self, closed: IntervalClosedType) -> Self:
    if closed not in VALID_CLOSED:
        msg = f"invalid option for 'closed': {closed}"
        raise ValueError(msg)
    left, right = (self._left, self._right)
    dtype = IntervalDtype(left.dtype, closed=closed)
    return self._simple_new(left, right, dtype=dtype)