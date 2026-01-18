import collections
import copy
import html
import itertools
import logging
import time
import warnings
from typing import (
import numpy as np
import ray
import ray.cloudpickle as pickle
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray._private.usage import usage_lib
from ray.air.util.tensor_extensions.utils import _create_possibly_ragged_ndarray
from ray.data._internal.block_list import BlockList
from ray.data._internal.compute import ComputeStrategy, TaskPoolStrategy
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.equalize import _equalize
from ray.data._internal.execution.interfaces import RefBundle
from ray.data._internal.execution.legacy_compat import _block_list_to_bundles
from ray.data._internal.iterator.iterator_impl import DataIteratorImpl
from ray.data._internal.iterator.stream_split_iterator import StreamSplitDataIterator
from ray.data._internal.lazy_block_list import LazyBlockList
from ray.data._internal.logical.operators.all_to_all_operator import (
from ray.data._internal.logical.operators.input_data_operator import InputData
from ray.data._internal.logical.operators.map_operator import (
from ray.data._internal.logical.operators.n_ary_operator import (
from ray.data._internal.logical.operators.n_ary_operator import Zip
from ray.data._internal.logical.operators.one_to_one_operator import Limit
from ray.data._internal.logical.operators.write_operator import Write
from ray.data._internal.logical.optimizers import LogicalPlan
from ray.data._internal.pandas_block import PandasBlockSchema
from ray.data._internal.plan import ExecutionPlan, OneToOneStage
from ray.data._internal.planner.plan_udf_map_op import (
from ray.data._internal.planner.plan_write_op import generate_write_fn
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.sort import SortKey
from ray.data._internal.split import _get_num_rows, _split_at_indices
from ray.data._internal.stage_impl import (
from ray.data._internal.stats import DatasetStats, DatasetStatsSummary, StatsManager
from ray.data._internal.util import (
from ray.data.aggregate import AggregateFn, Max, Mean, Min, Std, Sum
from ray.data.block import (
from ray.data.context import DataContext
from ray.data.datasource import (
from ray.data.iterator import DataIterator
from ray.data.random_access_dataset import RandomAccessDataset
from ray.types import ObjectRef
from ray.util.annotations import Deprecated, DeveloperAPI, PublicAPI
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.widgets import Template
from ray.widgets.util import repr_with_fallback
@ConsumptionAPI
def split_proportionately(self, proportions: List[float]) -> List['MaterializedDataset']:
    """Materialize and split the dataset using proportions.

        A common use case for this is splitting the dataset into train
        and test sets (equivalent to eg. scikit-learn's ``train_test_split``).
        For a higher level abstraction, see :meth:`Dataset.train_test_split`.

        This method splits datasets so that all splits
        always contains at least one row. If that isn't possible,
        an exception is raised.

        This is equivalent to caulculating the indices manually and calling
        :meth:`Dataset.split_at_indices`.

        Examples:
            >>> import ray
            >>> ds = ray.data.range(10)
            >>> d1, d2, d3 = ds.split_proportionately([0.2, 0.5])
            >>> d1.take_batch()
            {'id': array([0, 1])}
            >>> d2.take_batch()
            {'id': array([2, 3, 4, 5, 6])}
            >>> d3.take_batch()
            {'id': array([7, 8, 9])}

        Time complexity: O(num splits)

        Args:
            proportions: List of proportions to split the dataset according to.
                Must sum up to less than 1, and each proportion must be bigger
                than 0.

        Returns:
            The dataset splits.

        .. seealso::

            :meth:`Dataset.split`
                Unlike :meth:`~Dataset.split_proportionately`, which lets you split a
                dataset into different sizes, :meth:`Dataset.split` splits a dataset
                into approximately equal splits.

            :meth:`Dataset.split_at_indices`
                :meth:`Dataset.split_proportionately` uses this method under the hood.

            :meth:`Dataset.streaming_split`.
                Unlike :meth:`~Dataset.split`, :meth:`~Dataset.streaming_split`
                doesn't materialize the dataset in memory.
        """
    if len(proportions) < 1:
        raise ValueError('proportions must be at least of length 1')
    if sum(proportions) >= 1:
        raise ValueError('proportions must sum to less than 1')
    if any((p <= 0 for p in proportions)):
        raise ValueError('proportions must be bigger than 0')
    dataset_length = self.count()
    cumulative_proportions = np.cumsum(proportions)
    split_indices = [int(dataset_length * proportion) for proportion in cumulative_proportions]
    subtract = 0
    for i in range(len(split_indices) - 2, -1, -1):
        split_indices[i] -= subtract
        if split_indices[i] == split_indices[i + 1]:
            subtract += 1
            split_indices[i] -= 1
    if any((i <= 0 for i in split_indices)):
        raise ValueError("Couldn't create non-empty splits with the given proportions.")
    return self.split_at_indices(split_indices)