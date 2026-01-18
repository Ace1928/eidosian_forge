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
def compute_aligned_columns(*dfs, initial_columns=None, by=None):
    """Take row partitions, filter empty ones, and return joined columns for them."""
    if align_result_columns:
        valid_dfs = [df for df in dfs if not df.attrs.get(skip_on_aligning_flag, False)]
        if len(valid_dfs) == 0 and len(dfs) != 0:
            valid_dfs = dfs
        combined_cols = pandas.concat([df.iloc[:0] for df in valid_dfs], axis=0, join='outer').columns
    else:
        combined_cols = dfs[0].columns
    masks = None
    if add_missing_cats:
        masks, combined_cols = add_missing_categories_to_groupby(dfs, by, operator, initial_columns, combined_cols, is_udf_agg=align_result_columns, kwargs=kwargs.copy(), initial_dtypes=original_dtypes)
    return (combined_cols, masks) if align_result_columns else (None, masks)