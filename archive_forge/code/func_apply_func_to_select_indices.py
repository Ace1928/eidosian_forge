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
def apply_func_to_select_indices(cls, axis, partitions, func, indices, keep_remaining=False):
    """
        Apply a function to select indices.

        Parameters
        ----------
        axis : {0, 1}
            Axis to apply the `func` over.
        partitions : np.ndarray
            The partitions to which the `func` will apply.
        func : callable
            The function to apply to these indices of partitions.
        indices : dict
            The indices to apply the function to.
        keep_remaining : bool, default: False
            Whether or not to keep the other partitions. Some operations
            may want to drop the remaining partitions and keep
            only the results.

        Returns
        -------
        np.ndarray
            A NumPy array with partitions.

        Notes
        -----
        Your internal function must take a kwarg `internal_indices` for
        this to work correctly. This prevents information leakage of the
        internal index to the external representation.
        """
    if partitions.size == 0:
        return np.array([[]])
    if isinstance(func, dict):
        dict_func = func
    else:
        dict_func = None
    if not axis:
        partitions_for_apply = partitions.T
    else:
        partitions_for_apply = partitions
    if dict_func is not None:
        if not keep_remaining:
            result = np.array([cls._apply_func_to_list_of_partitions(func, partitions_for_apply[o_idx], func_dict={i_idx: dict_func[i_idx] for i_idx in list_to_apply if i_idx >= 0}) for o_idx, list_to_apply in indices.items()])
        else:
            result = np.array([partitions_for_apply[i] if i not in indices else cls._apply_func_to_list_of_partitions(func, partitions_for_apply[i], func_dict={idx: dict_func[idx] for idx in indices[i] if idx >= 0}) for i in range(len(partitions_for_apply))])
    elif not keep_remaining:
        result = np.array([cls._apply_func_to_list_of_partitions(func, partitions_for_apply[idx], internal_indices=list_to_apply) for idx, list_to_apply in indices.items()])
    else:
        result = np.array([partitions_for_apply[i] if i not in indices else cls._apply_func_to_list_of_partitions(func, partitions_for_apply[i], internal_indices=indices[i]) for i in range(len(partitions_for_apply))])
    return result.T if not axis else result