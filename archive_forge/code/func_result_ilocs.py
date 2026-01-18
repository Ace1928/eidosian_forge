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
def result_ilocs(self) -> npt.NDArray[np.intp]:
    """
        Get the original integer locations of result_index in the input.
        """
    group_index = get_group_index(self.codes, self.shape, sort=self._sort, xnull=True)
    group_index, _ = compress_group_index(group_index, sort=self._sort)
    if self.has_dropped_na:
        mask = np.where(group_index >= 0)
        null_gaps = np.cumsum(group_index == -1)[mask]
        group_index = group_index[mask]
    result = get_group_index_sorter(group_index, self.ngroups)
    if self.has_dropped_na:
        result += np.take(null_gaps, result)
    return result