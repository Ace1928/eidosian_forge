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
@PublicAPI
@wrap_auto_init
def read_datasource(datasource: Datasource, *, parallelism: int=-1, ray_remote_args: Dict[str, Any]=None, **read_args) -> Dataset:
    """Read a stream from a custom :class:`~ray.data.Datasource`.

    Args:
        datasource: The :class:`~ray.data.Datasource` to read data from.
        parallelism: The requested parallelism of the read. Parallelism might be
            limited by the available partitioning of the datasource. If set to -1,
            parallelism is automatically chosen based on the available cluster
            resources and estimated in-memory data size.
        read_args: Additional kwargs to pass to the :class:`~ray.data.Datasource`
            implementation.
        ray_remote_args: kwargs passed to :meth:`ray.remote` in the read tasks.

    Returns:
        :class:`~ray.data.Dataset` that reads data from the :class:`~ray.data.Datasource`.
    """
    ctx = DataContext.get_current()
    if ray_remote_args is None:
        ray_remote_args = {}
    if not datasource.supports_distributed_reads:
        ray_remote_args['scheduling_strategy'] = NodeAffinitySchedulingStrategy(ray.get_runtime_context().get_node_id(), soft=False)
    if 'scheduling_strategy' not in ray_remote_args:
        ray_remote_args['scheduling_strategy'] = ctx.scheduling_strategy
    force_local = False
    pa_ds = _lazy_import_pyarrow_dataset()
    if pa_ds:
        partitioning = read_args.get('dataset_kwargs', {}).get('partitioning', None)
        if isinstance(partitioning, pa_ds.Partitioning):
            logger.info(f'Forcing local metadata resolution since the provided partitioning {partitioning} is not serializable.')
            force_local = True
    if force_local:
        datasource_or_legacy_reader = _get_datasource_or_legacy_reader(datasource, ctx, read_args)
    else:
        scheduling_strategy = NodeAffinitySchedulingStrategy(ray.get_runtime_context().get_node_id(), soft=False)
        get_datasource_or_legacy_reader = cached_remote_fn(_get_datasource_or_legacy_reader, retry_exceptions=False, num_cpus=0).options(scheduling_strategy=scheduling_strategy)
        datasource_or_legacy_reader = ray.get(get_datasource_or_legacy_reader.remote(datasource, ctx, _wrap_arrow_serialization_workaround(read_args)))
    cur_pg = ray.util.get_current_placement_group()
    requested_parallelism, _, _, inmemory_size = _autodetect_parallelism(parallelism, ctx.target_max_block_size, DataContext.get_current(), datasource_or_legacy_reader, placement_group=cur_pg)
    read_tasks = datasource_or_legacy_reader.get_read_tasks(requested_parallelism)
    if not ctx.use_streaming_executor:
        _warn_on_high_parallelism(requested_parallelism, len(read_tasks))
    read_stage_name = f'Read{datasource.get_name()}'
    block_list = LazyBlockList(read_tasks, read_stage_name=read_stage_name, ray_remote_args=ray_remote_args, owned_by_consumer=False)
    block_list._estimated_num_blocks = len(read_tasks) if read_tasks else 0
    read_op = Read(datasource, datasource_or_legacy_reader, parallelism, inmemory_size, ray_remote_args)
    logical_plan = LogicalPlan(read_op)
    return Dataset(plan=ExecutionPlan(block_list, block_list.stats(), run_by_consumer=False), logical_plan=logical_plan)