from __future__ import annotations
from collections.abc import Sequence
from typing import (
import warnings
import numpy as np
from numpy import ma
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas._typing import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import NumpyEADtype
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import isna
import pandas.core.common as com
def range_to_ndarray(rng: range) -> np.ndarray:
    """
    Cast a range object to ndarray.
    """
    try:
        arr = np.arange(rng.start, rng.stop, rng.step, dtype='int64')
    except OverflowError:
        if rng.start >= 0 and rng.step > 0 or rng.step < 0 <= rng.stop:
            try:
                arr = np.arange(rng.start, rng.stop, rng.step, dtype='uint64')
            except OverflowError:
                arr = construct_1d_object_array_from_listlike(list(rng))
        else:
            arr = construct_1d_object_array_from_listlike(list(rng))
    return arr