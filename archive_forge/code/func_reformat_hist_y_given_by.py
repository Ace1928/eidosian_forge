from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from pandas.core.dtypes.missing import remove_na_arraylike
from pandas import (
from pandas.plotting._matplotlib.misc import unpack_single_str_list
def reformat_hist_y_given_by(y: np.ndarray, by: IndexLabel | None) -> np.ndarray:
    """Internal function to reformat y given `by` is applied or not for hist plot.

    If by is None, input y is 1-d with NaN removed; and if by is not None, groupby
    will take place and input y is multi-dimensional array.
    """
    if by is not None and len(y.shape) > 1:
        return np.array([remove_na_arraylike(col) for col in y.T]).T
    return remove_na_arraylike(y)