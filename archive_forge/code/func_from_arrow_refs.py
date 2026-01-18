import collections
import logging
import os
from typing import (
import numpy as np
import ray
from ray._private.auto_init_hook import wrap_auto_init
from ray.air.util.tensor_extensions.utils import _create_possibly_ragged_ndarray
from ray.data._internal.block_list import BlockList
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.lazy_block_list import LazyBlockList
from ray.data._internal.logical.operators.from_operators import (
from ray.data._internal.logical.operators.read_operator import Read
from ray.data._internal.logical.optimizers import LogicalPlan
from ray.data._internal.plan import ExecutionPlan
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.stats import DatasetStats
from ray.data._internal.util import (
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata
from ray.data.context import DataContext
from ray.data.dataset import Dataset, MaterializedDataset
from ray.data.datasource import (
from ray.data.datasource._default_metadata_providers import (
from ray.data.datasource.datasource import Reader
from ray.data.datasource.file_based_datasource import (
from ray.data.datasource.partitioning import Partitioning
from ray.types import ObjectRef
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
@DeveloperAPI
def from_arrow_refs(tables: Union[ObjectRef[Union['pyarrow.Table', bytes]], List[ObjectRef[Union['pyarrow.Table', bytes]]]]) -> MaterializedDataset:
    """Create a :class:`~ray.data.Dataset` from a list of Ray object references to
    PyArrow tables.

    Examples:
        >>> import pyarrow as pa
        >>> import ray
        >>> table_ref = ray.put(pa.table({"x": [1]}))
        >>> ray.data.from_arrow_refs(table_ref)
        MaterializedDataset(num_blocks=1, num_rows=1, schema={x: int64})

        Create a Ray Dataset from a list of PyArrow table references

        >>> ray.data.from_arrow_refs([table_ref, table_ref])
        MaterializedDataset(num_blocks=2, num_rows=2, schema={x: int64})


    Args:
        tables: A Ray object reference to Arrow table, or list of Ray object
                references to Arrow tables, or its streaming format in bytes.

    Returns:
         :class:`~ray.data.Dataset` holding data read from the tables.
    """
    if isinstance(tables, ray.ObjectRef):
        tables = [tables]
    get_metadata = cached_remote_fn(get_table_block_metadata)
    metadata = ray.get([get_metadata.remote(t) for t in tables])
    logical_plan = LogicalPlan(FromArrow(tables, metadata))
    return MaterializedDataset(ExecutionPlan(BlockList(tables, metadata, owned_by_consumer=False), DatasetStats(stages={'FromArrow': metadata}, parent=None), run_by_consumer=False), logical_plan)