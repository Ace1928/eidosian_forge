import copy
import os
from functools import partial
from itertools import groupby
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple, TypeVar, Union
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types
from . import config
from .utils.logging import get_logger
def fast_gather(self, indices: Union[List[int], np.ndarray]) -> pa.Table:
    """
        Create a pa.Table by gathering the records at the records at the specified indices. Should be faster
        than pa.concat_tables(table.fast_slice(int(i) % table.num_rows, 1) for i in indices) since NumPy can compute
        the binary searches in parallel, highly optimized C
        """
    if not len(indices):
        raise ValueError('Indices must be non-empty')
    batch_indices = np.searchsorted(self._offsets, indices, side='right') - 1
    return pa.Table.from_batches([self._batches[batch_idx].slice(i - self._offsets[batch_idx], 1) for batch_idx, i in zip(batch_indices, indices)], schema=self._schema)