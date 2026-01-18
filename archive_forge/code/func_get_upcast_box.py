import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def get_upcast_box(left, right, is_cmp: bool=False):
    """
    Get the box to use for 'expected' in an arithmetic or comparison operation.

    Parameters
    left : Any
    right : Any
    is_cmp : bool, default False
        Whether the operation is a comparison method.
    """
    if isinstance(left, DataFrame) or isinstance(right, DataFrame):
        return DataFrame
    if isinstance(left, Series) or isinstance(right, Series):
        if is_cmp and isinstance(left, Index):
            return np.array
        return Series
    if isinstance(left, Index) or isinstance(right, Index):
        if is_cmp:
            return np.array
        return Index
    return tm.to_array