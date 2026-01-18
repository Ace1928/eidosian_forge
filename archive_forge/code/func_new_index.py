from __future__ import annotations
import itertools
from typing import (
import warnings
import numpy as np
import pandas._libs.reshape as libreshape
from pandas.errors import PerformanceWarning
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.missing import notna
import pandas.core.algorithms as algos
from pandas.core.algorithms import (
from pandas.core.arrays.categorical import factorize_from_iterable
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.frame import DataFrame
from pandas.core.indexes.api import (
from pandas.core.reshape.concat import concat
from pandas.core.series import Series
from pandas.core.sorting import (
@cache_readonly
def new_index(self) -> MultiIndex:
    result_codes = [lab.take(self.compressor) for lab in self.sorted_labels[:-1]]
    if len(self.new_index_levels) == 1:
        level, level_codes = (self.new_index_levels[0], result_codes[0])
        if (level_codes == -1).any():
            level = level.insert(len(level), level._na_value)
        return level.take(level_codes).rename(self.new_index_names[0])
    return MultiIndex(levels=self.new_index_levels, codes=result_codes, names=self.new_index_names, verify_integrity=False)