from __future__ import annotations
from collections import abc
from typing import (
import warnings
import numpy as np
from pandas._config import using_copy_on_write
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import isna
from pandas.core.arrays.categorical import (
import pandas.core.common as com
from pandas.core.indexes.api import (
from pandas.core.internals import concatenate_managers
@cache_readonly
def new_axes(self) -> list[Index]:
    ndim = self._get_result_dim()
    return [self._get_concat_axis if i == self.bm_axis else self._get_comb_axis(i) for i in range(ndim)]