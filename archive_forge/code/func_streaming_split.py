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
def streaming_split(self, n: int, *, equal: bool=False, locality_hints: Optional[List['NodeIdStr']]=None) -> List[DataIterator]:
    """Returns ``n`` :class:`DataIterators <ray.data.DataIterator>` that can
        be used to read disjoint subsets of the dataset in parallel.

        This method is the recommended way to consume :class:`Datasets <Dataset>` for
        distributed training.

        Streaming split works by delegating the execution of this :class:`Dataset` to a
        coordinator actor. The coordinator pulls block references from the executed
        stream, and divides those blocks among ``n`` output iterators. Iterators pull
        blocks from the coordinator actor to return to their caller on ``next``.

        The returned iterators are also repeatable; each iteration will trigger a
        new execution of the Dataset. There is an implicit barrier at the start of
        each iteration, which means that `next` must be called on all iterators before
        the iteration starts.

        .. warning::

            Because iterators are pulling blocks from the same :class:`Dataset`
            execution, if one iterator falls behind, other iterators may be stalled.

        Examples:

            .. testcode::

                import ray

                ds = ray.data.range(100)
                it1, it2 = ds.streaming_split(2, equal=True)

            Consume data from iterators in parallel.

            .. testcode::

                @ray.remote
                def consume(it):
                    for batch in it.iter_batches():
                       pass

                ray.get([consume.remote(it1), consume.remote(it2)])

            You can loop over the iterators multiple times (multiple epochs).

            .. testcode::

                @ray.remote
                def train(it):
                    NUM_EPOCHS = 2
                    for _ in range(NUM_EPOCHS):
                        for batch in it.iter_batches():
                            pass

                ray.get([train.remote(it1), train.remote(it2)])

            The following remote function call blocks waiting for a read on ``it2`` to
            start.

            .. testcode::
                :skipif: True

                ray.get(train.remote(it1))

        Args:
            n: Number of output iterators to return.
            equal: If ``True``, each output iterator sees an exactly equal number
                of rows, dropping data if necessary. If ``False``, some iterators may
                see slightly more or less rows than others, but no data is dropped.
            locality_hints: Specify the node ids corresponding to each iterator
                location. Dataset will try to minimize data movement based on the
                iterator output locations. This list must have length ``n``. You can
                get the current node id of a task or actor by calling
                ``ray.get_runtime_context().get_node_id()``.

        Returns:
            The output iterator splits. These iterators are Ray-serializable and can
            be freely passed to any Ray task or actor.

        .. seealso::

            :meth:`Dataset.split`
                Unlike :meth:`~Dataset.streaming_split`, :meth:`~Dataset.split`
                materializes the dataset in memory.
        """
    return StreamSplitDataIterator.create(self, n, equal, locality_hints)