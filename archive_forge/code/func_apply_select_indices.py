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
def apply_select_indices(self, axis, func, apply_indices=None, row_labels=None, col_labels=None, new_index=None, new_columns=None, new_dtypes=None, keep_remaining=False, item_to_distribute=no_default):
    """
        Apply a function for a subset of the data.

        Parameters
        ----------
        axis : {0, 1}
            The axis to apply over.
        func : callable
            The function to apply.
        apply_indices : list-like, default: None
            The labels to apply over. Must be given if axis is provided.
        row_labels : list-like, default: None
            The row labels to apply over. Must be provided with
            `col_labels` to apply over both axes.
        col_labels : list-like, default: None
            The column labels to apply over. Must be provided
            with `row_labels` to apply over both axes.
        new_index : list-like, optional
            The index of the result, if known in advance.
        new_columns : list-like, optional
            The columns of the result, if known in advance.
        new_dtypes : pandas.Series, optional
            The dtypes of the result, if known in advance.
        keep_remaining : boolean, default: False
            Whether or not to drop the data that is not computed over.
        item_to_distribute : np.ndarray or scalar, default: no_default
            The item to split up so it can be applied over both axes.

        Returns
        -------
        PandasDataframe
            A new dataframe.
        """
    if new_index is None:
        new_index = self.index if axis == 1 else None
    if new_columns is None:
        new_columns = self.columns if axis == 0 else None
    if new_columns is not None and isinstance(new_dtypes, pandas.Series):
        assert new_dtypes.index.equals(new_columns), f"new_dtypes={new_dtypes!r} doesn't have the same columns as in new_columns={new_columns!r}"
    if axis is not None:
        assert apply_indices is not None
        old_index = self.index if axis else self.columns
        numeric_indices = old_index.get_indexer_for(apply_indices)
        dict_indices = self._get_dict_of_block_index(axis ^ 1, numeric_indices)
        new_partitions = self._partition_mgr_cls.apply_func_to_select_indices(axis, self._partitions, func, dict_indices, keep_remaining=keep_remaining)
        lengths_objs = {axis: [len(apply_indices)] if not keep_remaining else [self.row_lengths, self.column_widths][axis], axis ^ 1: [self.row_lengths, self.column_widths][axis ^ 1]}
        return self.__constructor__(new_partitions, new_index, new_columns, lengths_objs[0], lengths_objs[1], new_dtypes)
    else:
        assert row_labels is not None and col_labels is not None
        assert keep_remaining
        assert item_to_distribute is not no_default
        row_partitions_list = self._get_dict_of_block_index(0, row_labels).items()
        col_partitions_list = self._get_dict_of_block_index(1, col_labels).items()
        new_partitions = self._partition_mgr_cls.apply_func_to_indices_both_axis(self._partitions, func, row_partitions_list, col_partitions_list, item_to_distribute, self._row_lengths_cache, self._column_widths_cache)
        return self.__constructor__(new_partitions, new_index, new_columns, self._row_lengths_cache, self._column_widths_cache, new_dtypes)