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
@DeveloperAPI
def serialize_lineage(self) -> bytes:
    """
        Serialize this dataset's lineage, not the actual data or the existing data
        futures, to bytes that can be stored and later deserialized, possibly on a
        different cluster.

        Note that this will drop all computed data, and that everything is
        recomputed from scratch after deserialization.

        Use :py:meth:`Dataset.deserialize_lineage` to deserialize the serialized
        bytes returned from this method into a Dataset.

        .. note::
            Unioned and zipped datasets, produced by :py:meth`Dataset.union` and
            :py:meth:`Dataset.zip`, are not lineage-serializable.

        Examples:

            .. testcode::

                import ray

                ds = ray.data.read_csv("s3://anonymous@ray-example-data/iris.csv")
                serialized_ds = ds.serialize_lineage()
                ds = ray.data.Dataset.deserialize_lineage(serialized_ds)
                print(ds)

            .. testoutput::

                Dataset(
                   num_blocks=...,
                   num_rows=150,
                   schema={
                      sepal length (cm): double,
                      sepal width (cm): double,
                      petal length (cm): double,
                      petal width (cm): double,
                      target: int64
                   }
                )


        Returns:
            Serialized bytes containing the lineage of this dataset.
        """
    if not self.has_serializable_lineage():
        raise ValueError('Lineage-based serialization is not supported for this stream, which means that it cannot be used as a tunable hyperparameter. Lineage-based serialization is explicitly NOT supported for unioned or zipped datasets (see docstrings for those methods), and is only supported for datasets created from data that we know will still exist at deserialization time, e.g. external data in persistent cloud object stores or in-memory data from long-lived clusters. Concretely, all ray.data.read_*() APIs should support lineage-based serialization, while all of the ray.data.from_*() APIs do not. To allow this stream to be serialized to storage, write the data to an external store (such as AWS S3, GCS, or Azure Blob Storage) using the Dataset.write_*() APIs, and serialize a new dataset reading from the external store using the ray.data.read_*() APIs.')
    plan_copy = self._plan.deep_copy()
    logical_plan_copy = copy.copy(self._plan._logical_plan)
    ds = Dataset(plan_copy, logical_plan_copy)
    ds._plan.clear_block_refs()
    ds._set_uuid(self._get_uuid())

    def _reduce_remote_fn(rf: ray.remote_function.RemoteFunction):
        reconstructor, args, state = rf.__reduce__()
        state['_last_export_session_and_job'] = None
        return (reconstructor, args, state)
    context = ray._private.worker.global_worker.get_serialization_context()
    try:
        context._register_cloudpickle_reducer(ray.remote_function.RemoteFunction, _reduce_remote_fn)
        serialized = pickle.dumps(ds)
    finally:
        context._unregister_cloudpickle_reducer(ray.remote_function.RemoteFunction)
    return serialized