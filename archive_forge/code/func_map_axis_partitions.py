import os
import warnings
from abc import ABC
from functools import wraps
from typing import TYPE_CHECKING
import numpy as np
import pandas
from pandas._libs.lib import no_default
from modin.config import (
from modin.core.dataframe.pandas.utils import create_pandas_df_from_partitions
from modin.core.storage_formats.pandas.utils import compute_chunksize
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
@classmethod
def map_axis_partitions(cls, axis, partitions, map_func, keep_partitioning=False, num_splits=None, lengths=None, enumerate_partitions=False, **kwargs):
    """
        Apply `map_func` to every partition in `partitions` along given `axis`.

        Parameters
        ----------
        axis : {0, 1}
            Axis to perform the map across (0 - index, 1 - columns).
        partitions : NumPy 2D array
            Partitions of Modin Frame.
        map_func : callable
            Function to apply.
        keep_partitioning : boolean, default: False
            The flag to keep partition boundaries for Modin Frame if possible.
            Setting it to True disables shuffling data from one partition to another in case the resulting
            number of splits is equal to the initial number of splits.
        num_splits : int, optional
            The number of partitions to split the result into across the `axis`. If None, then the number
            of splits will be infered automatically. If `num_splits` is None and `keep_partitioning=True`
            then the number of splits is preserved.
        lengths : list of ints, default: None
            The list of lengths to shuffle the object. Note:
                1. Passing `lengths` omits the `num_splits` parameter as the number of splits
                will now be inferred from the number of integers present in `lengths`.
                2. When passing lengths you must explicitly specify `keep_partitioning=False`.
        enumerate_partitions : bool, default: False
            Whether or not to pass partition index into `map_func`.
            Note that `map_func` must be able to accept `partition_idx` kwarg.
        **kwargs : dict
            Additional options that could be used by different engines.

        Returns
        -------
        NumPy array
            An array of new partitions for Modin Frame.

        Notes
        -----
        This method should be used in the case when `map_func` relies on
        some global information about the axis.
        """
    return cls.broadcast_axis_partitions(axis=axis, left=partitions, apply_func=map_func, keep_partitioning=keep_partitioning, num_splits=num_splits, right=None, lengths=lengths, enumerate_partitions=enumerate_partitions, **kwargs)