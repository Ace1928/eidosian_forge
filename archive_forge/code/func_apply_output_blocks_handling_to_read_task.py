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
def apply_output_blocks_handling_to_read_task(read_task: ReadTask, additional_split_factor: Optional[int]):
    """Patch the read task and apply output blocks handling logic.
    This function is only used for compability with the legacy LazyBlockList code path.
    """
    transform_fns: List[MapTransformFn] = []
    transform_fns.append(BuildOutputBlocksMapTransformFn.for_blocks())
    if additional_split_factor is not None:
        transform_fns.append(ApplyAdditionalSplitToOutputBlocks(additional_split_factor))
    map_transformer = MapTransformer(transform_fns)
    ctx = DataContext.get_current()
    map_transformer.set_target_max_block_size(ctx.target_max_block_size)
    original_read_fn = read_task._read_fn

    def new_read_fn():
        blocks = original_read_fn()
        return map_transformer.apply_transform(blocks, None)
    read_task._read_fn = new_read_fn