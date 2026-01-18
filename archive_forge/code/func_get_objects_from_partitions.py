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
def get_objects_from_partitions(cls, partitions):
    """
        Get the objects wrapped by `partitions` (in parallel if supported).

        Parameters
        ----------
        partitions : np.ndarray
            NumPy array with ``PandasDataframePartition``-s.

        Returns
        -------
        list
            The objects wrapped by `partitions`.
        """
    if hasattr(cls, '_execution_wrapper'):
        for idx, part in enumerate(partitions):
            if hasattr(part, 'force_materialization'):
                partitions[idx] = part.force_materialization()
        assert all([len(partition.list_of_blocks) == 1 for partition in partitions]), 'Implementation assumes that each partition contains a single block.'
        return cls._execution_wrapper.materialize([partition.list_of_blocks[0] for partition in partitions])
    return [partition.get() for partition in partitions]