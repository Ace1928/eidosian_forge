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
def column_partitions(cls, partitions, full_axis=True):
    """
        Get the list of `BaseDataframeAxisPartition` objects representing column-wise partitions.

        Parameters
        ----------
        partitions : list-like
            List of (smaller) partitions to be combined to column-wise partitions.
        full_axis : bool, default: True
            Whether or not this partition contains the entire column axis.

        Returns
        -------
        list
            A list of `BaseDataframeAxisPartition` objects.

        Notes
        -----
        Each value in this list will be an `BaseDataframeAxisPartition` object.
        `BaseDataframeAxisPartition` is located in `axis_partition.py`.
        """
    if not isinstance(partitions, list):
        partitions = [partitions]
    return [cls._column_partitions_class(col, full_axis=full_axis) for frame in partitions for col in frame.T]