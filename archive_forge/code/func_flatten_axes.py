from __future__ import annotations
from math import ceil
from typing import TYPE_CHECKING
import warnings
from matplotlib import ticker
import matplotlib.table
import numpy as np
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.generic import (
def flatten_axes(axes: Axes | Sequence[Axes]) -> np.ndarray:
    if not is_list_like(axes):
        return np.array([axes])
    elif isinstance(axes, (np.ndarray, ABCIndex)):
        return np.asarray(axes).ravel()
    return np.array(axes)