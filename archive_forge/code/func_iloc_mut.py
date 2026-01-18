import ast
import hashlib
import re
import warnings
from collections.abc import Iterable
from typing import Hashable, List
import numpy as np
import pandas
from pandas._libs import lib
from pandas.api.types import is_scalar
from pandas.core.apply import reconstruct_func
from pandas.core.common import is_bool_indexer
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import (
from pandas.core.groupby.base import transformation_kernels
from pandas.core.indexes.api import ensure_index_from_sequences
from pandas.core.indexing import check_bool_indexer
from pandas.errors import DataError
from modin.config import CpuCount, RangePartitioning, use_range_partitioning_groupby
from modin.core.dataframe.algebra import (
from modin.core.dataframe.algebra.default2pandas.groupby import (
from modin.core.dataframe.pandas.metadata import (
from modin.core.storage_formats import BaseQueryCompiler
from modin.error_message import ErrorMessage
from modin.logging import get_logger
from modin.utils import (
from .aggregations import CorrCovBuilder
from .groupby import GroupbyReduceImpl, PivotTableImpl
from .merge import MergeImpl
from .utils import get_group_names, merge_partitioning
def iloc_mut(partition, row_internal_indices, col_internal_indices, item):
    """
            Write `value` in a specified location in a single partition.

            Parameters
            ----------
            partition : pandas.DataFrame
                Partition of the self frame.
            row_internal_indices : list of ints
                Positional indices of rows in this particular partition
                to write `item` to.
            col_internal_indices : list of ints
                Positional indices of columns in this particular partition
                to write `item` to.
            item : 2D-array
                Value to write.

            Returns
            -------
            pandas.DataFrame
                Partition data with updated values.
            """
    partition = partition.copy()
    try:
        partition.iloc[row_internal_indices, col_internal_indices] = item
    except ValueError:
        partition.iloc[row_internal_indices, col_internal_indices] = item.copy()
    return partition