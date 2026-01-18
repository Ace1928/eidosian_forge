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
def lazy_map_partitions(cls, partitions, map_func, func_args=None, func_kwargs=None, enumerate_partitions=False):
    """
        Apply `map_func` to every partition in `partitions` *lazily*.

        Parameters
        ----------
        partitions : NumPy 2D array
            Partitions of Modin Frame.
        map_func : callable
            Function to apply.
        func_args : iterable, optional
            Positional arguments for the 'map_func'.
        func_kwargs : dict, optional
            Keyword arguments for the 'map_func'.
        enumerate_partitions : bool, default: False

        Returns
        -------
        NumPy array
            An array of partitions
        """
    preprocessed_map_func = cls.preprocess_func(map_func)
    return np.array([[part.add_to_apply_calls(preprocessed_map_func, *(tuple() if func_args is None else func_args), **func_kwargs if func_kwargs is not None else {}, **{'partition_idx': i} if enumerate_partitions else {}) for part in row] for i, row in enumerate(partitions)])