from typing import Any, Iterator, Tuple
import ray
from ray.data._internal.block_list import BlockList
from ray.data._internal.compute import ActorPoolStrategy, get_compute
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.operators.base_physical_operator import (
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.operators.limit_operator import LimitOperator
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.operators.map_transformer import (
from ray.data._internal.execution.util import make_callable_class_concurrent
from ray.data._internal.lazy_block_list import LazyBlockList
from ray.data._internal.logical.interfaces.logical_plan import LogicalPlan
from ray.data._internal.logical.operators.read_operator import Read
from ray.data._internal.logical.optimizers import get_execution_plan
from ray.data._internal.logical.rules.set_read_parallelism import (
from ray.data._internal.logical.util import record_operators_usage
from ray.data._internal.memory_tracing import trace_allocation
from ray.data._internal.plan import AllToAllStage, ExecutionPlan, OneToOneStage, Stage
from ray.data._internal.planner.plan_read_op import (
from ray.data._internal.stage_impl import LimitStage, RandomizeBlocksStage
from ray.data._internal.stats import DatasetStats, StatsDict
from ray.data.block import Block, BlockMetadata, CallableClass, List
from ray.data.context import DataContext
from ray.data.datasource import ReadTask
from ray.types import ObjectRef
def execute_to_legacy_block_iterator(executor: Executor, plan: ExecutionPlan, allow_clear_input_blocks: bool, dataset_uuid: str) -> Iterator[Tuple[ObjectRef[Block], BlockMetadata]]:
    """Same as execute_to_legacy_bundle_iterator but returning blocks and metadata."""
    bundle_iter = execute_to_legacy_bundle_iterator(executor, plan, allow_clear_input_blocks, dataset_uuid)
    for bundle in bundle_iter:
        for block, metadata in bundle.blocks:
            yield (block, metadata)