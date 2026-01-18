from __future__ import annotations
from collections.abc import (
from functools import wraps
from sys import getsizeof
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import (
from pandas._libs.hashtable import duplicated
from pandas._typing import (
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import coerce_indexer_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import is_array_like
from pandas.core.dtypes.missing import (
import pandas.core.algorithms as algos
from pandas.core.array_algos.putmask import validate_putmask
from pandas.core.arrays import (
from pandas.core.arrays.categorical import (
import pandas.core.common as com
from pandas.core.construction import sanitize_array
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import (
from pandas.core.indexes.frozen import FrozenList
from pandas.core.ops.invalid import make_invalid_op
from pandas.core.sorting import (
from pandas.io.formats.printing import (
@cache_readonly
def is_monotonic_increasing(self) -> bool:
    """
        Return a boolean if the values are equal or increasing.
        """
    if any((-1 in code for code in self.codes)):
        return False
    if all((level.is_monotonic_increasing for level in self.levels)):
        return libalgos.is_lexsorted([x.astype('int64', copy=False) for x in self.codes])
    values = [self._get_level_values(i)._values for i in reversed(range(len(self.levels)))]
    try:
        sort_order = np.lexsort(values)
        return Index(sort_order).is_monotonic_increasing
    except TypeError:
        return Index(self._values).is_monotonic_increasing