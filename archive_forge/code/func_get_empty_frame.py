from __future__ import annotations
from collections import defaultdict
from collections.abc import (
import itertools
from typing import (
import numpy as np
from pandas._libs.sparse import IntIndex
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.arrays import SparseArray
from pandas.core.arrays.categorical import factorize_from_iterable
from pandas.core.arrays.string_ import StringDtype
from pandas.core.frame import DataFrame
from pandas.core.indexes.api import (
from pandas.core.series import Series
def get_empty_frame(data) -> DataFrame:
    index: Index | np.ndarray
    if isinstance(data, Series):
        index = data.index
    else:
        index = default_index(len(data))
    return DataFrame(index=index)