import abc
from typing import TYPE_CHECKING, Dict, List, Union
import numpy as np
import pandas
import pyarrow as pa
from pandas.core.dtypes.common import is_string_dtype
from modin.pandas.indexing import is_range_like
from modin.utils import _inherit_docstrings
from .dataframe.utils import EMPTY_ARROW_TABLE, ColNameCodec, get_common_arrow_type
from .db_worker import DbTable
from .expr import InputRefExpr, LiteralExpr, OpExpr
def collect_partitions(self):
    """
        Collect all partitions participating in a tree.

        Returns
        -------
        list
            A list of collected partitions.
        """
    partitions = []
    self.walk_dfs(lambda a, b: a._append_partitions(b), partitions)
    return partitions