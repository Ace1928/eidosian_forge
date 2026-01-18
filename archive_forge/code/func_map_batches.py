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
def map_batches(self, fn: UserDefinedFunction[DataBatch, DataBatch], *, batch_size: Union[int, None, Literal['default']]='default', compute: Optional[ComputeStrategy]=None, batch_format: Optional[str]='default', zero_copy_batch: bool=False, fn_args: Optional[Iterable[Any]]=None, fn_kwargs: Optional[Dict[str, Any]]=None, fn_constructor_args: Optional[Iterable[Any]]=None, fn_constructor_kwargs: Optional[Dict[str, Any]]=None, num_cpus: Optional[float]=None, num_gpus: Optional[float]=None, concurrency: Optional[Union[int, Tuple[int, int]]]=None, **ray_remote_args) -> 'Dataset':
    """Apply the given function to batches of data.

        This method is useful for preprocessing data and performing inference. To learn
        more, see :ref:`Transforming batches <transforming_batches>`.

        You can use either a function or a callable class to perform the transformation.
        For functions, Ray Data uses stateless Ray tasks. For classes, Ray Data uses
        stateful Ray actors. For more information, see
        :ref:`Stateful Transforms <stateful_transforms>`.

        .. tip::
            If ``fn`` doesn't mutate its input, set ``zero_copy_batch=True`` to improve
            performance and decrease memory utilization.

        Examples:

            Call :meth:`~Dataset.map_batches` to transform your data.

            .. testcode::

                from typing import Dict
                import numpy as np
                import ray

                def add_dog_years(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
                    batch["age_in_dog_years"] = 7 * batch["age"]
                    return batch

                ds = (
                    ray.data.from_items([
                        {"name": "Luna", "age": 4},
                        {"name": "Rory", "age": 14},
                        {"name": "Scout", "age": 9},
                    ])
                    .map_batches(add_dog_years)
                )
                ds.show()

            .. testoutput::

                {'name': 'Luna', 'age': 4, 'age_in_dog_years': 28}
                {'name': 'Rory', 'age': 14, 'age_in_dog_years': 98}
                {'name': 'Scout', 'age': 9, 'age_in_dog_years': 63}

            If your function returns large objects, yield outputs in chunks.

            .. testcode::

                from typing import Dict
                import ray
                import numpy as np

                def map_fn_with_large_output(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
                    for i in range(3):
                        yield {"large_output": np.ones((100, 1000))}

                ds = (
                    ray.data.from_items([1])
                    .map_batches(map_fn_with_large_output)
                )

            If you require stateful transfomation,
            use Python callable class. Here is an example showing how to use stateful transforms to create model inference workers, without having to reload the model on each call.

            .. testcode::

                from typing import Dict
                import numpy as np
                import torch
                import ray

                class TorchPredictor:

                    def __init__(self):
                        self.model = torch.nn.Identity().cuda()
                        self.model.eval()

                    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
                        inputs = torch.as_tensor(batch["data"], dtype=torch.float32).cuda()
                        with torch.inference_mode():
                            batch["output"] = self.model(inputs).detach().cpu().numpy()
                        return batch

                ds = (
                    ray.data.from_numpy(np.ones((32, 100)))
                    .map_batches(
                        TorchPredictor,
                        # Two workers with one GPU each
                        concurrency=2,
                        # Batch size is required if you're using GPUs.
                        batch_size=4,
                        num_gpus=1
                    )
                )

            To learn more, see
            :ref:`End-to-end: Offline Batch Inference <batch_inference_home>`.

        Args:
            fn: The function or generator to apply to a record batch, or a class type
                that can be instantiated to create such a callable. Note ``fn`` must be
                pickle-able.
            batch_size: The desired number of rows in each batch, or ``None`` to use
                entire blocks as batches (blocks may contain different numbers of rows).
                The actual size of the batch provided to ``fn`` may be smaller than
                ``batch_size`` if ``batch_size`` doesn't evenly divide the block(s) sent
                to a given map task. Default batch_size is 1024 with "default".
            compute: This argument is deprecated. Use ``concurrency`` argument.
            batch_format: If ``"default"`` or ``"numpy"``, batches are
                ``Dict[str, numpy.ndarray]``. If ``"pandas"``, batches are
                ``pandas.DataFrame``.
            zero_copy_batch: Whether ``fn`` should be provided zero-copy, read-only
                batches. If this is ``True`` and no copy is required for the
                ``batch_format`` conversion, the batch is a zero-copy, read-only
                view on data in Ray's object store, which can decrease memory
                utilization and improve performance. If this is ``False``, the batch
                is writable, which requires an extra copy to guarantee.
                If ``fn`` mutates its input, this needs to be ``False`` in order to
                avoid "assignment destination is read-only" or "buffer source array is
                read-only" errors. Default is ``False``.
            fn_args: Positional arguments to pass to ``fn`` after the first argument.
                These arguments are top-level arguments to the underlying Ray task.
            fn_kwargs: Keyword arguments to pass to ``fn``. These arguments are
                top-level arguments to the underlying Ray task.
            fn_constructor_args: Positional arguments to pass to ``fn``'s constructor.
                You can only provide this if ``fn`` is a callable class. These arguments
                are top-level arguments in the underlying Ray actor construction task.
            fn_constructor_kwargs: Keyword arguments to pass to ``fn``'s constructor.
                This can only be provided if ``fn`` is a callable class. These arguments
                are top-level arguments in the underlying Ray actor construction task.
            num_cpus: The number of CPUs to reserve for each parallel map worker.
            num_gpus: The number of GPUs to reserve for each parallel map worker. For
                example, specify `num_gpus=1` to request 1 GPU for each parallel map worker.
            concurrency: The number of Ray workers to use concurrently. For a fixed-sized
                worker pool of size ``n``, specify ``concurrency=n``. For an autoscaling
                worker pool from ``m`` to ``n`` workers, specify ``concurrency=(m, n)``.
            ray_remote_args: Additional resource requirements to request from
                ray for each map worker.

        .. note::

            The size of the batches provided to ``fn`` might be smaller than the
            specified ``batch_size`` if ``batch_size`` doesn't evenly divide the
            block(s) sent to a given map task.

        .. seealso::

            :meth:`~Dataset.iter_batches`
                Call this function to iterate over batches of data.

            :meth:`~Dataset.flat_map`
                Call this method to create new records from existing ones. Unlike
                :meth:`~Dataset.map`, a function passed to :meth:`~Dataset.flat_map`
                can return multiple records.

            :meth:`~Dataset.map`
                Call this method to transform one record at time.

        """
    compute = get_compute_strategy(fn, fn_constructor_args=fn_constructor_args, compute=compute, concurrency=concurrency)
    if num_cpus is not None:
        ray_remote_args['num_cpus'] = num_cpus
    if num_gpus is not None:
        ray_remote_args['num_gpus'] = num_gpus
    batch_format = _apply_strict_mode_batch_format(batch_format)
    if batch_format == 'native':
        logger.warning("The 'native' batch format has been renamed 'default'.")
    min_rows_per_block = None
    if batch_size is not None and batch_size != 'default':
        if batch_size < 1:
            raise ValueError('Batch size cannot be negative or 0')
        min_rows_per_block = batch_size
    batch_size = _apply_strict_mode_batch_size(batch_size, use_gpu='num_gpus' in ray_remote_args)
    if batch_format not in VALID_BATCH_FORMATS:
        raise ValueError(f'The batch format must be one of {VALID_BATCH_FORMATS}, got: {batch_format}')
    ctx = DataContext.get_current()
    transform_fn = generate_map_batches_fn(target_max_block_size=ctx.target_max_block_size, batch_size=batch_size, batch_format=batch_format, zero_copy_batch=zero_copy_batch)
    if hasattr(fn, '__self__') and isinstance(fn.__self__, ray.data.preprocessor.Preprocessor):
        stage_name = fn.__self__.__class__.__name__
    else:
        stage_name = f'MapBatches({getattr(fn, '__name__', type(fn))})'
    stage = OneToOneStage(stage_name, transform_fn, compute, ray_remote_args, min_rows_per_block=min_rows_per_block, fn=fn, fn_args=fn_args, fn_kwargs=fn_kwargs, fn_constructor_args=fn_constructor_args, fn_constructor_kwargs=fn_constructor_kwargs)
    plan = self._plan.with_stage(stage)
    logical_plan = self._logical_plan
    if logical_plan is not None:
        map_batches_op = MapBatches(logical_plan.dag, fn, batch_size=batch_size, batch_format=batch_format, zero_copy_batch=zero_copy_batch, min_rows_per_block=min_rows_per_block, fn_args=fn_args, fn_kwargs=fn_kwargs, fn_constructor_args=fn_constructor_args, fn_constructor_kwargs=fn_constructor_kwargs, compute=compute, ray_remote_args=ray_remote_args)
        logical_plan = LogicalPlan(map_batches_op)
    return Dataset(plan, logical_plan)