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
def n_ary_op(self, op, right_frames: list, join_type='outer', copartition_along_columns=True, labels='replace', dtypes=None):
    """
        Perform an n-opary operation by joining with other Modin DataFrame(s).

        Parameters
        ----------
        op : callable
            Function to apply after the join.
        right_frames : list of PandasDataframe
            Modin DataFrames to join with.
        join_type : str, default: "outer"
            Type of join to apply.
        copartition_along_columns : bool, default: True
            Whether to perform copartitioning along columns or not.
            For some ops this isn't needed (e.g., `fillna`).
        labels : {"replace", "drop"}, default: "replace"
            Whether use labels from joined DataFrame or drop altogether to make
            them be computed lazily later.
        dtypes : series, default: None
            Dtypes of the resultant dataframe, this argument will be
            received if the resultant dtypes of n-opary operation is precomputed.

        Returns
        -------
        PandasDataframe
            New Modin DataFrame.
        """
    left_parts, list_of_right_parts, joined_index, row_lengths = self._copartition(0, right_frames, join_type, sort=True)
    if copartition_along_columns:
        new_left_frame = self.__constructor__(left_parts, joined_index, self.copy_columns_cache(copy_lengths=True), row_lengths, self._column_widths_cache)
        new_right_frames = [self.__constructor__(right_parts, joined_index, right_frame.copy_columns_cache(copy_lengths=True), row_lengths, right_frame._column_widths_cache) for right_parts, right_frame in zip(list_of_right_parts, right_frames)]
        left_parts, list_of_right_parts, joined_columns, column_widths = new_left_frame._copartition(1, new_right_frames, join_type, sort=True)
    else:
        joined_columns = self.copy_columns_cache(copy_lengths=True)
        column_widths = self._column_widths_cache
    new_frame = np.array([]) if len(left_parts) == 0 or any((len(right_parts) == 0 for right_parts in list_of_right_parts)) else self._partition_mgr_cls.n_ary_operation(left_parts, op, list_of_right_parts)
    if labels == 'drop':
        joined_index = joined_columns = row_lengths = column_widths = None
    return self.__constructor__(new_frame, joined_index, joined_columns, row_lengths, column_widths, dtypes)