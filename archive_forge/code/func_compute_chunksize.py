from __future__ import annotations
import re
from math import ceil
from typing import Generator, Hashable, List, Optional
import numpy as np
import pandas
from modin.config import MinPartitionSize, NPartitions
def compute_chunksize(axis_len: int, num_splits: int, min_block_size: int) -> int:
    """
    Compute the number of elements (rows/columns) to include in each partition.

    Chunksize is defined the same for both axes.

    Parameters
    ----------
    axis_len : int
        Element count in an axis.
    num_splits : int
        The number of splits.
    min_block_size : int
        Minimum number of rows/columns in a single split.

    Returns
    -------
    int
        Integer number of rows/columns to split the DataFrame will be returned.
    """
    if not isinstance(min_block_size, int) or min_block_size <= 0:
        raise ValueError(f"'min_block_size' should be int > 0, passed: min_block_size={min_block_size!r}")
    chunksize = axis_len // num_splits
    if axis_len % num_splits:
        chunksize += 1
    return max(chunksize, min_block_size)