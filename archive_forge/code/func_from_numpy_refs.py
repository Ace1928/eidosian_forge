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
def from_numpy_refs(ndarrays: Union[ObjectRef[np.ndarray], List[ObjectRef[np.ndarray]]]) -> MaterializedDataset:
    """Creates a :class:`~ray.data.Dataset` from a list of Ray object references to
    NumPy ndarrays.

    Examples:
        >>> import numpy as np
        >>> import ray
        >>> arr_ref = ray.put(np.array([1]))
        >>> ray.data.from_numpy_refs(arr_ref)
        MaterializedDataset(num_blocks=1, num_rows=1, schema={data: int64})

        Create a Ray Dataset from a list of NumPy array references.

        >>> ray.data.from_numpy_refs([arr_ref, arr_ref])
        MaterializedDataset(num_blocks=2, num_rows=2, schema={data: int64})

    Args:
        ndarrays: A Ray object reference to a NumPy ndarray or a list of Ray object
            references to NumPy ndarrays.

    Returns:
        :class:`~ray.data.Dataset` holding data from the given ndarrays.
    """
    if isinstance(ndarrays, ray.ObjectRef):
        ndarrays = [ndarrays]
    elif isinstance(ndarrays, list):
        for ndarray in ndarrays:
            if not isinstance(ndarray, ray.ObjectRef):
                raise ValueError(f'Expected list of Ray object refs, got list containing {type(ndarray)}')
    else:
        raise ValueError(f'Expected Ray object ref or list of Ray object refs, got {type(ndarray)}')
    ctx = DataContext.get_current()
    ndarray_to_block_remote = cached_remote_fn(ndarray_to_block, num_returns=2)
    res = [ndarray_to_block_remote.remote(ndarray, ctx) for ndarray in ndarrays]
    blocks, metadata = map(list, zip(*res))
    metadata = ray.get(metadata)
    logical_plan = LogicalPlan(FromNumpy(blocks, metadata))
    return MaterializedDataset(ExecutionPlan(BlockList(blocks, metadata, owned_by_consumer=False), DatasetStats(stages={'FromNumpy': metadata}, parent=None), run_by_consumer=False), logical_plan)