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
@wait_computations_if_benchmark_mode
def broadcast_axis_partitions(cls, axis, apply_func, left, right, keep_partitioning=False, num_splits=None, apply_indices=None, enumerate_partitions=False, lengths=None, apply_func_args=None, **kwargs):
    """
        Broadcast the `right` partitions to `left` and apply `apply_func` along full `axis`.

        Parameters
        ----------
        axis : {0, 1}
            Axis to apply and broadcast over.
        apply_func : callable
            Function to apply.
        left : NumPy 2D array
            Left partitions.
        right : NumPy 2D array
            Right partitions.
        keep_partitioning : boolean, default: False
            The flag to keep partition boundaries for Modin Frame if possible.
            Setting it to True disables shuffling data from one partition to another in case the resulting
            number of splits is equal to the initial number of splits.
        num_splits : int, optional
            The number of partitions to split the result into across the `axis`. If None, then the number
            of splits will be infered automatically. If `num_splits` is None and `keep_partitioning=True`
            then the number of splits is preserved.
        apply_indices : list of ints, default: None
            Indices of `axis ^ 1` to apply function over.
        enumerate_partitions : bool, default: False
            Whether or not to pass partition index into `apply_func`.
            Note that `apply_func` must be able to accept `partition_idx` kwarg.
        lengths : list of ints, default: None
            The list of lengths to shuffle the object. Note:
                1. Passing `lengths` omits the `num_splits` parameter as the number of splits
                will now be inferred from the number of integers present in `lengths`.
                2. When passing lengths you must explicitly specify `keep_partitioning=False`.
        apply_func_args : list-like, optional
            Positional arguments to pass to the `func`.
        **kwargs : dict
            Additional options that could be used by different engines.

        Returns
        -------
        NumPy array
            An array of partition objects.
        """
    ErrorMessage.catch_bugs_and_request_email(failure_condition=keep_partitioning and lengths is not None, extra_log=f'`keep_partitioning` must be set to `False` when passing `lengths`. Got: keep_partitioning={keep_partitioning!r} | lengths={lengths!r}')
    if keep_partitioning and num_splits is None:
        num_splits = len(left) if axis == 0 else len(left.T)
    elif lengths:
        num_splits = len(lengths)
    elif num_splits is None:
        num_splits = NPartitions.get()
    else:
        ErrorMessage.catch_bugs_and_request_email(failure_condition=not isinstance(num_splits, int), extra_log=f'Expected `num_splits` to be an integer, got: {type(num_splits)} | num_splits={num_splits!r}')
    preprocessed_map_func = cls.preprocess_func(apply_func)
    left_partitions = cls.axis_partition(left, axis)
    right_partitions = None if right is None else cls.axis_partition(right, axis)
    kw = {'num_splits': num_splits, 'other_axis_partition': right_partitions, 'maintain_partitioning': keep_partitioning}
    if lengths:
        kw['lengths'] = lengths
        kw['manual_partition'] = True
    if apply_indices is None:
        apply_indices = np.arange(len(left_partitions))
    result_blocks = np.array([left_partitions[i].apply(preprocessed_map_func, *(apply_func_args if apply_func_args else []), **kw, **{'partition_idx': idx} if enumerate_partitions else {}, **kwargs) for idx, i in enumerate(apply_indices)])
    return result_blocks.T if not axis else result_blocks