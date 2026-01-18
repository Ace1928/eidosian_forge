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
@ConsumptionAPI(pattern='store memory.', insert_after=True)
def materialize(self) -> 'MaterializedDataset':
    """Execute and materialize this dataset into object store memory.

        This can be used to read all blocks into memory. By default, Dataset
        doesn't read blocks from the datasource until the first transform.

        Note that this does not mutate the original Dataset. Only the blocks of the
        returned MaterializedDataset class are pinned in memory.

        Examples:
            >>> import ray
            >>> ds = ray.data.range(10)
            >>> materialized_ds = ds.materialize()
            >>> materialized_ds
            MaterializedDataset(num_blocks=..., num_rows=10, schema={id: int64})

        Returns:
            A MaterializedDataset holding the materialized data blocks.
        """
    copy = Dataset.copy(self, _deep_copy=True, _as=MaterializedDataset)
    copy._plan.execute(force_read=True)
    blocks = copy._plan._snapshot_blocks
    blocks_with_metadata = blocks.get_blocks_with_metadata() if blocks else []
    ref_bundles = [RefBundle(blocks=[block_with_metadata], owns_blocks=False) for block_with_metadata in blocks_with_metadata]
    logical_plan = LogicalPlan(InputData(input_data=ref_bundles))
    output = MaterializedDataset(ExecutionPlan(blocks, copy._plan.stats(), run_by_consumer=False), logical_plan)
    output._plan.execute()
    output._set_uuid(copy._get_uuid())
    return output