from __future__ import annotations
import copy
from datetime import timedelta
from functools import partial
import inspect
from textwrap import dedent
from typing import (
import numpy as np
from pandas._libs.tslibs import (
import pandas._libs.window.aggregations as window_aggregations
from pandas.compat._optional import import_optional_dependency
from pandas.errors import DataError
from pandas.util._decorators import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import notna
from pandas.core._numba import executor
from pandas.core.algorithms import factorize
from pandas.core.apply import ResamplerWindowApply
from pandas.core.arrays import ExtensionArray
from pandas.core.base import SelectionMixin
import pandas.core.common as com
from pandas.core.indexers.objects import (
from pandas.core.indexes.api import (
from pandas.core.reshape.concat import concat
from pandas.core.util.numba_ import (
from pandas.core.window.common import (
from pandas.core.window.doc import (
from pandas.core.window.numba_ import (
from pandas.core.arrays.datetimelike import dtype_to_unit
def corr_func(x, y):
    x_array = self._prep_values(x)
    y_array = self._prep_values(y)
    window_indexer = self._get_window_indexer()
    min_periods = self.min_periods if self.min_periods is not None else window_indexer.window_size
    start, end = window_indexer.get_window_bounds(num_values=len(x_array), min_periods=min_periods, center=self.center, closed=self.closed, step=self.step)
    self._check_window_bounds(start, end, len(x_array))
    with np.errstate(all='ignore'):
        mean_x_y = window_aggregations.roll_mean(x_array * y_array, start, end, min_periods)
        mean_x = window_aggregations.roll_mean(x_array, start, end, min_periods)
        mean_y = window_aggregations.roll_mean(y_array, start, end, min_periods)
        count_x_y = window_aggregations.roll_sum(notna(x_array + y_array).astype(np.float64), start, end, 0)
        x_var = window_aggregations.roll_var(x_array, start, end, min_periods, ddof)
        y_var = window_aggregations.roll_var(y_array, start, end, min_periods, ddof)
        numerator = (mean_x_y - mean_x * mean_y) * (count_x_y / (count_x_y - ddof))
        denominator = (x_var * y_var) ** 0.5
        result = numerator / denominator
    return Series(result, index=x.index, name=x.name, copy=False)