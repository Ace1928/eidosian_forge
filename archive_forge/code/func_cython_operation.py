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
def cython_operation(self, *, values: ArrayLike, axis: AxisInt, min_count: int=-1, comp_ids: np.ndarray, ngroups: int, **kwargs) -> ArrayLike:
    """
        Call our cython function, with appropriate pre- and post- processing.
        """
    self._validate_axis(axis, values)
    if not isinstance(values, np.ndarray):
        return values._groupby_op(how=self.how, has_dropped_na=self.has_dropped_na, min_count=min_count, ngroups=ngroups, ids=comp_ids, **kwargs)
    return self._cython_op_ndim_compat(values, min_count=min_count, ngroups=ngroups, comp_ids=comp_ids, mask=None, **kwargs)