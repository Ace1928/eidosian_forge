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
def from_spark(df: 'pyspark.sql.DataFrame', *, parallelism: Optional[int]=None) -> MaterializedDataset:
    """Create a :class:`~ray.data.Dataset` from a
    `Spark DataFrame <https://spark.apache.org/docs/3.1.1/api/python/reference/api/pyspark.sql.DataFrame.html>`_.

    Args:
        df: A `Spark DataFrame`_, which must be created by RayDP (Spark-on-Ray).
        parallelism: The amount of parallelism to use for the dataset. If
            not provided, the parallelism is equal to the number of partitions of
            the original Spark DataFrame.

    Returns:
        A :class:`~ray.data.MaterializedDataset` holding rows read from the DataFrame.
    """
    import raydp
    return raydp.spark.spark_dataframe_to_ray_dataset(df, parallelism)