from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from ._internal.table_block import TableBlockAccessor
from ray.data._internal import sort
from ray.data._internal.compute import ComputeStrategy
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.logical.interfaces import LogicalPlan
from ray.data._internal.logical.operators.all_to_all_operator import Aggregate
from ray.data._internal.plan import AllToAllStage
from ray.data._internal.push_based_shuffle import PushBasedShufflePlan
from ray.data._internal.shuffle import ShuffleOp, SimpleShufflePlan
from ray.data._internal.sort import SortKey
from ray.data.aggregate import AggregateFn, Count, Max, Mean, Min, Std, Sum
from ray.data.aggregate._aggregate import _AggregateOnKeyBase
from ray.data.block import (
from ray.data.context import DataContext
from ray.data.dataset import DataBatch, Dataset
from ray.util.annotations import PublicAPI
def get_key_boundaries(block_accessor: BlockAccessor) -> List[int]:
    """Compute block boundaries based on the key(s)"""
    import numpy as np
    keys = block_accessor.to_numpy(self._key)
    if isinstance(keys, dict):
        convert_to_multi_column_sorted_key = np.vectorize(_MultiColumnSortedKey)
        keys: np.ndarray = convert_to_multi_column_sorted_key(*keys.values())
    boundaries = []
    start = 0
    while start < keys.size:
        end = start + np.searchsorted(keys[start:], keys[start], side='right')
        boundaries.append(end)
        start = end
    return boundaries