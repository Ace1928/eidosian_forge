import datetime
import re
from typing import TYPE_CHECKING, Callable, Dict, Hashable, List, Optional, Union
import numpy as np
import pandas
from pandas._libs.lib import no_default
from pandas.api.types import is_object_dtype
from pandas.core.dtypes.common import is_dtype_equal, is_list_like, is_numeric_dtype
from pandas.core.indexes.api import Index, RangeIndex
from modin.config import Engine, IsRayCluster, MinPartitionSize, NPartitions
from modin.core.dataframe.base.dataframe.dataframe import ModinDataframe
from modin.core.dataframe.base.dataframe.utils import Axis, JoinType, is_trivial_index
from modin.core.dataframe.pandas.dataframe.utils import (
from modin.core.dataframe.pandas.metadata import (
from modin.core.storage_formats.pandas.parsers import (
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.core.storage_formats.pandas.utils import get_length_list
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from modin.pandas.indexing import is_range_like
from modin.pandas.utils import check_both_not_none, is_full_grab_slice
from modin.utils import MODIN_UNNAMED_SERIES_LABEL
@lazy_metadata_decorator(apply_axis='both')
def apply_full_axis(self, axis, func, new_index=None, new_columns=None, apply_indices=None, enumerate_partitions: bool=False, dtypes=None, keep_partitioning=True, num_splits=None, sync_labels=True, pass_axis_lengths_to_partitions=False):
    """
        Perform a function across an entire axis.

        Parameters
        ----------
        axis : {0, 1}
            The axis to apply over (0 - rows, 1 - columns).
        func : callable
            The function to apply.
        new_index : list-like, optional
            The index of the result. We may know this in advance,
            and if not provided it must be computed.
        new_columns : list-like, optional
            The columns of the result. We may know this in
            advance, and if not provided it must be computed.
        apply_indices : list-like, default: None
            Indices of `axis ^ 1` to apply function over.
        enumerate_partitions : bool, default: False
            Whether pass partition index into applied `func` or not.
            Note that `func` must be able to obtain `partition_idx` kwarg.
        dtypes : list-like, optional
            The data types of the result. This is an optimization
            because there are functions that always result in a particular data
            type, and allows us to avoid (re)computing it.
        keep_partitioning : boolean, default: True
            The flag to keep partition boundaries for Modin Frame if possible.
            Setting it to True disables shuffling data from one partition to another in case the resulting
            number of splits is equal to the initial number of splits.
        num_splits : int, optional
            The number of partitions to split the result into across the `axis`. If None, then the number
            of splits will be infered automatically. If `num_splits` is None and `keep_partitioning=True`
            then the number of splits is preserved.
        sync_labels : boolean, default: True
            Synchronize external indexes (`new_index`, `new_columns`) with internal indexes.
            This could be used when you're certain that the indices in partitions are equal to
            the provided hints in order to save time on syncing them.
        pass_axis_lengths_to_partitions : bool, default: False
            Whether pass partition lengths along `axis ^ 1` to the kernel `func`.
            Note that `func` must be able to obtain `df, *axis_lengths`.

        Returns
        -------
        PandasDataframe
            A new dataframe.

        Notes
        -----
        The data shape may change as a result of the function.
        """
    return self.broadcast_apply_full_axis(axis=axis, func=func, new_index=new_index, new_columns=new_columns, apply_indices=apply_indices, enumerate_partitions=enumerate_partitions, dtypes=dtypes, other=None, keep_partitioning=keep_partitioning, num_splits=num_splits, sync_labels=sync_labels, pass_axis_lengths_to_partitions=pass_axis_lengths_to_partitions)