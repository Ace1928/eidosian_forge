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
def broadcast_apply(self, axis, func, other, join_type='left', copartition=True, labels='keep', dtypes=None):
    """
        Broadcast axis partitions of `other` to partitions of `self` and apply a function.

        Parameters
        ----------
        axis : {0, 1}
            Axis to broadcast over.
        func : callable
            Function to apply.
        other : PandasDataframe
            Modin DataFrame to broadcast.
        join_type : str, default: "left"
            Type of join to apply.
        copartition : bool, default: True
            Whether to align indices/partitioning of the `self` and `other` frame.
            Disabling this may save some time, however, you have to be 100% sure that
            the indexing and partitioning are identical along the broadcasting axis,
            this might be the case for example if `other` is a projection of the `self`
            or vice-versa. If copartitioning is disabled and partitioning/indexing are
            incompatible then you may end up with undefined behavior.
        labels : {"keep", "replace", "drop"}, default: "keep"
            Whether keep labels from `self` Modin DataFrame, replace them with labels
            from joined DataFrame or drop altogether to make them be computed lazily later.
        dtypes : "copy", pandas.Series or None, default: None
            Dtypes of the result. "copy" to keep old dtypes and None to compute them on demand.

        Returns
        -------
        PandasDataframe
            New Modin DataFrame.
        """
    if copartition:
        left_parts, right_parts, joined_index, partition_sizes_along_axis = self._copartition(axis, other, join_type, sort=not self.get_axis(axis).equals(other.get_axis(axis)))
        right_parts = right_parts[0]
    else:
        left_parts = self._partitions
        right_parts = other._partitions
        partition_sizes_along_axis, joined_index = (self._get_axis_lengths_cache(axis), self.copy_axis_cache(axis))
    new_frame = self._partition_mgr_cls.broadcast_apply(axis, func, left_parts, right_parts)
    if isinstance(dtypes, str) and dtypes == 'copy':
        dtypes = self.copy_dtypes_cache()

    def _pick_axis(get_axis, sizes_cache):
        if labels == 'keep':
            return (get_axis(), sizes_cache)
        if labels == 'replace':
            return (joined_index, partition_sizes_along_axis)
        assert labels == 'drop', f'Unexpected `labels`: {labels}'
        return (None, None)
    if axis == 0:
        new_index, new_row_lengths = _pick_axis(self.copy_index_cache, self._row_lengths_cache)
        new_columns, new_column_widths = (self.copy_columns_cache(), self._column_widths_cache)
    else:
        new_index, new_row_lengths = (self.copy_index_cache(), self._row_lengths_cache)
        new_columns, new_column_widths = _pick_axis(self.copy_columns_cache, self._column_widths_cache)
    return self.__constructor__(new_frame, new_index, new_columns, new_row_lengths, new_column_widths, dtypes=dtypes)