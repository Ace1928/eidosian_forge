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
def from_dask(df: 'dask.dataframe.DataFrame') -> MaterializedDataset:
    """Create a :class:`~ray.data.Dataset` from a
    `Dask DataFrame <https://docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.html#dask.dataframe.DataFrame>`_.

    Args:
        df: A `Dask DataFrame`_.

    Returns:
        A :class:`~ray.data.MaterializedDataset` holding rows read from the DataFrame.
    """
    import dask
    from ray.util.dask import ray_dask_get
    partitions = df.to_delayed()
    persisted_partitions = dask.persist(*partitions, scheduler=ray_dask_get)
    import pandas

    def to_ref(df):
        if isinstance(df, pandas.DataFrame):
            return ray.put(df)
        elif isinstance(df, ray.ObjectRef):
            return df
        else:
            raise ValueError(f'Expected a Ray object ref or a Pandas DataFrame, got {type(df)}')
    ds = from_pandas_refs([to_ref(next(iter(part.dask.values()))) for part in persisted_partitions])
    return ds