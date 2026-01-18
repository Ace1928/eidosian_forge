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
def from_torch(dataset: 'torch.utils.data.Dataset') -> Dataset:
    """Create a :class:`~ray.data.Dataset` from a
    `Torch Dataset <https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset/>`_.

    .. note::
        The input dataset can either be map-style or iterable-style, and can have arbitrarily large amount of data.
        The data will be sequentially streamed with one single read task.

    Examples:
        >>> import ray
        >>> from torchvision import datasets
        >>> dataset = datasets.MNIST("data", download=True)  # doctest: +SKIP
        >>> ds = ray.data.from_torch(dataset)  # doctest: +SKIP
        >>> ds  # doctest: +SKIP
        MaterializedDataset(num_blocks=..., num_rows=60000, schema={item: object})
        >>> ds.take(1)  # doctest: +SKIP
        {"item": (<PIL.Image.Image image mode=L size=28x28 at 0x...>, 5)}

    Args:
        dataset: A `Torch Dataset`_.

    Returns:
        A :class:`~ray.data.Dataset` containing the Torch dataset samples.
    """
    ray_remote_args = {'scheduling_strategy': NodeAffinitySchedulingStrategy(ray.get_runtime_context().get_node_id(), soft=False)}
    return read_datasource(TorchDatasource(dataset=dataset), parallelism=1, ray_remote_args=ray_remote_args)