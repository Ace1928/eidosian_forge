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
def materialize_futures(cls, input_list):
    """
        Materialize all futures in the input list.

        Parameters
        ----------
        input_list : list
            The list that has to be manipulated.

        Returns
        -------
        list
           A new list with materialized objects.
        """
    if input_list is None:
        return None
    filtered_list = []
    filtered_idx = []
    for idx, item in enumerate(input_list):
        if cls._execution_wrapper.is_future(item):
            filtered_idx.append(idx)
            filtered_list.append(item)
    filtered_list = cls._execution_wrapper.materialize(filtered_list)
    result = input_list.copy()
    for idx, item in zip(filtered_idx, filtered_list):
        result[idx] = item
    return result