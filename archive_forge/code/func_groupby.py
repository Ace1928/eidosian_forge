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
@AllToAllAPI
def groupby(self, key: Union[str, List[str], None]) -> 'GroupedData':
    """Group rows of a :class:`Dataset` according to a column.

        Use this method to transform data based on a
        categorical variable.

        Examples:

            .. testcode::

                import pandas as pd
                import ray

                def normalize_variety(group: pd.DataFrame) -> pd.DataFrame:
                    for feature in group.drop("variety").columns:
                        group[feature] = group[feature] / group[feature].abs().max()
                    return group

                ds = (
                    ray.data.read_parquet("s3://anonymous@ray-example-data/iris.parquet")
                    .groupby("variety")
                    .map_groups(normalize_variety, batch_format="pandas")
                )

        Time complexity: O(dataset size * log(dataset size / parallelism))

        Args:
            key: A column name or list of column names.
            If this is ``None``, place all rows in a single group.

        Returns:
            A lazy :class:`~ray.data.grouped_data.GroupedData`.

        .. seealso::

            :meth:`~ray.data.grouped_data.GroupedData.map_groups`
                Call this method to transform groups of data.
        """
    from ray.data.grouped_data import GroupedData
    if key is not None:
        SortKey(key).validate_schema(self.schema(fetch_if_missing=True))
    return GroupedData(self, key)