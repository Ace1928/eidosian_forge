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
def broadcast_apply_select_indices(self, axis, func, other, apply_indices=None, numeric_indices=None, keep_remaining=False, broadcast_all=True, new_index=None, new_columns=None):
    """
        Apply a function to select indices at specified axis and broadcast partitions of `other` Modin DataFrame.

        Parameters
        ----------
        axis : {0, 1}
            Axis to apply function along.
        func : callable
            Function to apply.
        other : PandasDataframe
            Partitions of which should be broadcasted.
        apply_indices : list, default: None
            List of labels to apply (if `numeric_indices` are not specified).
        numeric_indices : list, default: None
            Numeric indices to apply (if `apply_indices` are not specified).
        keep_remaining : bool, default: False
            Whether drop the data that is not computed over or not.
        broadcast_all : bool, default: True
            Whether broadcast the whole axis of right frame to every
            partition or just a subset of it.
        new_index : pandas.Index, optional
            Index of the result. We may know this in advance,
            and if not provided it must be computed.
        new_columns : pandas.Index, optional
            Columns of the result. We may know this in advance,
            and if not provided it must be computed.

        Returns
        -------
        PandasDataframe
            New Modin DataFrame.
        """
    assert apply_indices is not None or numeric_indices is not None, 'Indices to apply must be specified!'
    if other is None:
        if apply_indices is None:
            apply_indices = self.get_axis(axis)[numeric_indices]
        return self.apply_select_indices(axis=axis, func=func, apply_indices=apply_indices, keep_remaining=keep_remaining, new_index=new_index, new_columns=new_columns)
    if numeric_indices is None:
        old_index = self.index if axis else self.columns
        numeric_indices = old_index.get_indexer_for(apply_indices)
    dict_indices = self._get_dict_of_block_index(axis ^ 1, numeric_indices)
    broadcasted_dict = other._prepare_frame_to_broadcast(axis, dict_indices, broadcast_all=broadcast_all)
    new_partitions = self._partition_mgr_cls.broadcast_apply_select_indices(axis, func, self._partitions, other._partitions, dict_indices, broadcasted_dict, keep_remaining)
    return self.__constructor__(new_partitions, index=new_index, columns=new_columns)