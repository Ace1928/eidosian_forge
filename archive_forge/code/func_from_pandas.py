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
def from_pandas(dfs: Union['pandas.DataFrame', List['pandas.DataFrame']]) -> MaterializedDataset:
    """Create a :class:`~ray.data.Dataset` from a list of pandas dataframes.

    Examples:
        >>> import pandas as pd
        >>> import ray
        >>> df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> ray.data.from_pandas(df)
        MaterializedDataset(num_blocks=1, num_rows=3, schema={a: int64, b: int64})

       Create a Ray Dataset from a list of Pandas DataFrames.

        >>> ray.data.from_pandas([df, df])
        MaterializedDataset(num_blocks=2, num_rows=6, schema={a: int64, b: int64})

    Args:
        dfs: A pandas dataframe or a list of pandas dataframes.

    Returns:
        :class:`~ray.data.Dataset` holding data read from the dataframes.
    """
    import pandas as pd
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
    from ray.air.util.data_batch_conversion import _cast_ndarray_columns_to_tensor_extension
    context = DataContext.get_current()
    if context.enable_tensor_extension_casting:
        dfs = [_cast_ndarray_columns_to_tensor_extension(df.copy()) for df in dfs]
    return from_pandas_refs([ray.put(df) for df in dfs])