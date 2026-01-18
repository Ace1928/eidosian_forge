import warnings
import numpy as np
import pandas
from modin.config import MinPartitionSize
from modin.core.dataframe.base.partitioning.axis_partition import (
from modin.core.storage_formats.pandas.utils import (
from .partition import PandasDataframePartition
@classmethod
def deploy_func_between_two_axis_partitions(cls, axis, func, f_args, f_kwargs, num_splits, len_of_left, other_shape, *partitions, min_block_size, return_generator=False):
    """
        Deploy a function along a full axis between two data sets.

        Parameters
        ----------
        axis : {0, 1}
            The axis to perform the function along.
        func : callable
            The function to perform.
        f_args : list or tuple
            Positional arguments to pass to ``func``.
        f_kwargs : dict
            Keyword arguments to pass to ``func``.
        num_splits : int
            The number of splits to return (see `split_result_of_axis_func_pandas`).
        len_of_left : int
            The number of values in `partitions` that belong to the left data set.
        other_shape : np.ndarray
            The shape of right frame in terms of partitions, i.e.
            (other_shape[i-1], other_shape[i]) will indicate slice to restore i-1 axis partition.
        *partitions : iterable
            All partitions that make up the full axis (row or column) for both data sets.
        min_block_size : int
            Minimum number of rows/columns in a single split.
        return_generator : bool, default: False
            Return a generator from the function, set to `True` for Ray backend
            as Ray remote functions can return Generators.

        Returns
        -------
        list | Generator
            A list or generator of pandas DataFrames.
        """
    lt_frame = pandas.concat(partitions[:len_of_left], axis=axis, copy=False)
    rt_parts = partitions[len_of_left:]
    del partitions
    combined_axis = [pandas.concat(rt_parts[other_shape[i - 1]:other_shape[i]], axis=axis, copy=False) for i in range(1, len(other_shape))]
    del rt_parts
    rt_frame = pandas.concat(combined_axis, axis=axis ^ 1, copy=False)
    del combined_axis
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        result = func(lt_frame, rt_frame, *f_args, **f_kwargs)
    del lt_frame, rt_frame
    if return_generator:
        return generate_result_of_axis_func_pandas(axis, num_splits, result, min_block_size)
    else:
        return split_result_of_axis_func_pandas(axis, num_splits, result, min_block_size)