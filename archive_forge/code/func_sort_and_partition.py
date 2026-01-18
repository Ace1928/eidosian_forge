import collections
import heapq
import random
from typing import (
import numpy as np
from ray._private.utils import _get_pyarrow_version
from ray.air.constants import TENSOR_COLUMN_NAME
from ray.data._internal.arrow_ops import transform_polars, transform_pyarrow
from ray.data._internal.numpy_support import (
from ray.data._internal.table_block import TableBlockAccessor, TableBlockBuilder
from ray.data._internal.util import _truncated_repr, find_partitions
from ray.data.block import (
from ray.data.context import DataContext
from ray.data.row import TableRow
def sort_and_partition(self, boundaries: List[T], sort_key: 'SortKey') -> List['Block']:
    if self._table.num_rows == 0:
        return [self._empty_table() for _ in range(len(boundaries) + 1)]
    context = DataContext.get_current()
    sort = get_sort_transform(context)
    table = sort(self._table, sort_key)
    if len(boundaries) == 0:
        return [table]
    return find_partitions(table, boundaries, sort_key)