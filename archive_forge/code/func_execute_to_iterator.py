import copy
import functools
import itertools
from typing import (
import ray
from ray._private.internal_api import get_memory_info_reply, get_state_from_address
from ray.data._internal.block_list import BlockList
from ray.data._internal.compute import (
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.lazy_block_list import LazyBlockList
from ray.data._internal.logical.operators.input_data_operator import InputData
from ray.data._internal.logical.operators.read_operator import Read
from ray.data._internal.logical.rules.operator_fusion import _are_remote_args_compatible
from ray.data._internal.logical.rules.set_read_parallelism import (
from ray.data._internal.planner.plan_read_op import (
from ray.data._internal.stats import DatasetStats, DatasetStatsSummary
from ray.data._internal.util import (
from ray.data.block import Block, BlockMetadata
from ray.data.context import DataContext
from ray.types import ObjectRef
from ray.util.debug import log_once
def execute_to_iterator(self, allow_clear_input_blocks: bool=True, force_read: bool=False) -> Tuple[Iterator[Tuple[ObjectRef[Block], BlockMetadata]], DatasetStats, Optional['Executor']]:
    """Execute this plan, returning an iterator.

        If the streaming execution backend is enabled, this will use streaming
        execution to generate outputs, otherwise it will fall back to bulk exec.

        Args:
            allow_clear_input_blocks: Whether we should try to clear the input blocks
                for each stage.
            force_read: Whether to force the read stage to fully execute.

        Returns:
            Tuple of iterator over output blocks and the executor.
        """
    ctx = self._context
    if not ctx.use_streaming_executor or self.has_computed_output():
        return (self.execute(allow_clear_input_blocks, force_read).iter_blocks_with_metadata(), self._snapshot_stats, None)
    from ray.data._internal.execution.legacy_compat import execute_to_legacy_block_iterator
    from ray.data._internal.execution.streaming_executor import StreamingExecutor
    metrics_tag = create_dataset_tag(self._dataset_name, self._dataset_uuid)
    executor = StreamingExecutor(copy.deepcopy(ctx.execution_options), metrics_tag)
    block_iter = execute_to_legacy_block_iterator(executor, self, allow_clear_input_blocks=allow_clear_input_blocks, dataset_uuid=self._dataset_uuid)
    gen = iter(block_iter)
    try:
        block_iter = itertools.chain([next(gen)], gen)
    except StopIteration:
        pass
    self._snapshot_stats = executor.get_stats()
    return (block_iter, self._snapshot_stats, executor)