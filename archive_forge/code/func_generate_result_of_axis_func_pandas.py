from __future__ import annotations
import re
from math import ceil
from typing import Generator, Hashable, List, Optional
import numpy as np
import pandas
from modin.config import MinPartitionSize, NPartitions
def generate_result_of_axis_func_pandas(axis: int, num_splits: int, result: pandas.DataFrame, min_block_size: int, length_list: Optional[list]=None) -> Generator:
    """
    Generate pandas DataFrame evenly based on the provided number of splits.

    Parameters
    ----------
    axis : {0, 1}
        Axis to split across. 0 means index axis when 1 means column axis.
    num_splits : int
        Number of splits to separate the DataFrame into.
        This parameter is ignored if `length_list` is specified.
    result : pandas.DataFrame
        DataFrame to split.
    min_block_size : int
        Minimum number of rows/columns in a single split.
    length_list : list of ints, optional
        List of slice lengths to split DataFrame into. This is used to
        return the DataFrame to its original partitioning schema.

    Yields
    ------
    Generator
        Generates 'num_splits' dataframes as a result of axis function.
    """
    if num_splits == 1:
        yield result
    else:
        if length_list is None:
            length_list = get_length_list(result.shape[axis], num_splits, min_block_size)
        length_list = np.insert(length_list, obj=0, values=[0])
        sums = np.cumsum(length_list)
        axis = 0 if isinstance(result, pandas.Series) else axis
        for i in range(len(sums) - 1):
            if axis == 0:
                chunk = result.iloc[sums[i]:sums[i + 1]]
            else:
                chunk = result.iloc[:, sums[i]:sums[i + 1]]
            if isinstance(chunk.axes[axis], pandas.MultiIndex):
                chunk = chunk.set_axis(chunk.axes[axis].remove_unused_levels(), axis=axis, copy=False)
            yield chunk