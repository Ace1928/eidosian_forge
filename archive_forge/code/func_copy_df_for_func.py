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
def copy_df_for_func(func, display_name: str=None):
    """
    Build function that execute specified `func` against passed frame inplace.

    Built function copies passed frame, applies `func` to the copy and returns
    the modified frame.

    Parameters
    ----------
    func : callable(pandas.DataFrame)
        The function, usually updates a dataframe inplace.
    display_name : str, optional
        The function's name, which is displayed by progress bar.

    Returns
    -------
    callable(pandas.DataFrame)
        A callable function to be applied in the partitions.
    """

    def caller(df, *args, **kwargs):
        """Apply specified function the passed frame inplace."""
        df = df.copy()
        func(df, *args, **kwargs)
        return df
    if display_name is not None:
        caller.__name__ = display_name
    return caller