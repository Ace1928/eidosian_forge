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
@staticmethod
def numpy_to_block(batch: Union[np.ndarray, Dict[str, np.ndarray], Dict[str, list]]) -> 'pyarrow.Table':
    import pyarrow as pa
    from ray.data.extensions.tensor_extension import ArrowTensorArray
    if isinstance(batch, np.ndarray):
        batch = {TENSOR_COLUMN_NAME: batch}
    elif not isinstance(batch, collections.abc.Mapping) or any((not is_valid_udf_return(col) for col in batch.values())):
        raise ValueError(f'Batch must be an ndarray or dictionary of ndarrays when converting a numpy batch to a block, got: {type(batch)} ({_truncated_repr(batch)})')
    new_batch = {}
    for col_name, col in batch.items():
        col = convert_udf_returns_to_numpy(col)
        if col.dtype.type is np.object_ or col.ndim > 1:
            col = ArrowTensorArray.from_numpy(col)
        new_batch[col_name] = col
    return pa.Table.from_pydict(new_batch)