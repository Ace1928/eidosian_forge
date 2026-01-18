from typing import Iterable, List, Optional
import ray
import ray.cloudpickle as cloudpickle
from ray.data._internal.execution.interfaces import PhysicalOperator, RefBundle
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.operators.map_transformer import (
from ray.data._internal.logical.operators.read_operator import Read
from ray.data._internal.util import _warn_on_high_parallelism, call_with_retry
from ray.data.block import Block
from ray.data.context import DataContext
from ray.data.datasource.datasource import ReadTask
def plan_read_op(op: Read) -> PhysicalOperator:
    """Get the corresponding DAG of physical operators for Read.

    Note this method only converts the given `op`, but not its input dependencies.
    See Planner.plan() for more details.
    """

    def get_input_data(target_max_block_size) -> List[RefBundle]:
        parallelism = op.get_detected_parallelism()
        assert parallelism is not None, 'Read parallelism must be set by the optimizer before execution'
        read_tasks = op._datasource_or_legacy_reader.get_read_tasks(parallelism)
        _warn_on_high_parallelism(parallelism, len(read_tasks))
        return [RefBundle([(ray.put(read_task), cleaned_metadata(read_task))], owns_blocks=False) for read_task in read_tasks]
    inputs = InputDataBuffer(input_data_factory=get_input_data)

    def do_read(blocks: Iterable[ReadTask], _: TaskContext) -> Iterable[Block]:
        """Yield from read tasks, with retry logic upon transient read errors."""
        for read_task in blocks:
            read_fn_name = read_task._read_fn.__name__
            yield from call_with_retry(f=read_task, match=READ_FILE_RETRY_ON_ERRORS, description=f'read file {read_fn_name}', max_attempts=READ_FILE_MAX_ATTEMPTS, max_backoff_s=READ_FILE_RETRY_MAX_BACKOFF_SECONDS)
    transform_fns: List[MapTransformFn] = [BlockMapTransformFn(do_read)]
    transform_fns.append(BuildOutputBlocksMapTransformFn.for_blocks())
    map_transformer = MapTransformer(transform_fns)
    return MapOperator.create(map_transformer, inputs, name=op.name, target_max_block_size=None, ray_remote_args=op._ray_remote_args)