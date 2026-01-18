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
def apply_full_axis_select_indices(self, axis, func, apply_indices=None, numeric_indices=None, new_index=None, new_columns=None, keep_remaining=False, new_dtypes=None):
    """
        Apply a function across an entire axis for a subset of the data.

        Parameters
        ----------
        axis : int
            The axis to apply over.
        func : callable
            The function to apply.
        apply_indices : list-like, default: None
            The labels to apply over.
        numeric_indices : list-like, default: None
            The indices to apply over.
        new_index : list-like, optional
            The index of the result. We may know this in advance,
            and if not provided it must be computed.
        new_columns : list-like, optional
            The columns of the result. We may know this in
            advance, and if not provided it must be computed.
        keep_remaining : boolean, default: False
            Whether or not to drop the data that is not computed over.
        new_dtypes : ModinDtypes or pandas.Series, optional
            The data types of the result. This is an optimization
            because there are functions that always result in a particular data
            type, and allows us to avoid (re)computing it.

        Returns
        -------
        PandasDataframe
            A new dataframe.
        """
    assert apply_indices is not None or numeric_indices is not None
    old_index = self.index if axis else self.columns
    if apply_indices is not None:
        numeric_indices = old_index.get_indexer_for(apply_indices)
    dict_indices = self._get_dict_of_block_index(axis ^ 1, numeric_indices)
    new_partitions = self._partition_mgr_cls.apply_func_to_select_indices_along_full_axis(axis, self._partitions, func, dict_indices, keep_remaining=keep_remaining)
    if new_index is None:
        new_index = self.index if axis == 1 else None
    if new_columns is None:
        new_columns = self.columns if axis == 0 else None
    return self.__constructor__(new_partitions, new_index, new_columns, None, None, dtypes=new_dtypes)