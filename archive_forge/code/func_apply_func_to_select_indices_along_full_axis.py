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
def apply_func_to_select_indices_along_full_axis(cls, axis, partitions, func, indices, keep_remaining=False):
    """
        Apply a function to a select subset of full columns/rows.

        Parameters
        ----------
        axis : {0, 1}
            The axis to apply the function over.
        partitions : np.ndarray
            The partitions to which the `func` will apply.
        func : callable
            The function to apply.
        indices : list-like
            The global indices to apply the func to.
        keep_remaining : bool, default: False
            Whether or not to keep the other partitions.
            Some operations may want to drop the remaining partitions and
            keep only the results.

        Returns
        -------
        np.ndarray
            A NumPy array with partitions.

        Notes
        -----
        This should be used when you need to apply a function that relies
        on some global information for the entire column/row, but only need
        to apply a function to a subset.
        For your func to operate directly on the indices provided,
        it must use `internal_indices` as a keyword argument.
        """
    if partitions.size == 0:
        return np.array([[]])
    if isinstance(func, dict):
        dict_func = func
    else:
        dict_func = None
    preprocessed_func = cls.preprocess_func(func)
    if not keep_remaining:
        selected_partitions = partitions.T if not axis else partitions
        selected_partitions = np.array([selected_partitions[i] for i in indices])
        selected_partitions = selected_partitions.T if not axis else selected_partitions
    else:
        selected_partitions = partitions
    if not axis:
        partitions_for_apply = cls.column_partitions(selected_partitions)
        partitions_for_remaining = partitions.T
    else:
        partitions_for_apply = cls.row_partitions(selected_partitions)
        partitions_for_remaining = partitions
    if dict_func is not None:
        if not keep_remaining:
            result = np.array([part.apply(preprocessed_func, func_dict={idx: dict_func[idx] for idx in indices[i]}) for i, part in zip(indices, partitions_for_apply)])
        else:
            result = np.array([partitions_for_remaining[i] if i not in indices else cls._apply_func_to_list_of_partitions(preprocessed_func, partitions_for_apply[i], func_dict={idx: dict_func[idx] for idx in indices[i]}) for i in range(len(partitions_for_apply))])
    elif not keep_remaining:
        result = np.array([part.apply(preprocessed_func, internal_indices=indices[i]) for i, part in zip(indices, partitions_for_apply)])
    else:
        result = np.array([partitions_for_remaining[i] if i not in indices else partitions_for_apply[i].apply(preprocessed_func, internal_indices=indices[i]) for i in range(len(partitions_for_remaining))])
    return result.T if not axis else result