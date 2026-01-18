from __future__ import annotations
import collections
import functools
from typing import (
import numpy as np
from pandas._libs import (
import pandas._libs.groupby as libgroupby
from pandas._typing import (
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import (
from pandas.core.frame import DataFrame
from pandas.core.groupby import grouper
from pandas.core.indexes.api import (
from pandas.core.series import Series
from pandas.core.sorting import (
@final
def apply_groupwise(self, f: Callable, data: DataFrame | Series, axis: AxisInt=0) -> tuple[list, bool]:
    mutated = False
    splitter = self._get_splitter(data, axis=axis)
    group_keys = self.group_keys_seq
    result_values = []
    zipped = zip(group_keys, splitter)
    for key, group in zipped:
        object.__setattr__(group, 'name', key)
        group_axes = group.axes
        res = f(group)
        if not mutated and (not _is_indexed_like(res, group_axes, axis)):
            mutated = True
        result_values.append(res)
    if len(group_keys) == 0 and getattr(f, '__name__', None) in ['skew', 'sum', 'prod']:
        f(data.iloc[:0])
    return (result_values, mutated)