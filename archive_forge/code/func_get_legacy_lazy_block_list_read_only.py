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
def get_legacy_lazy_block_list_read_only(plan: ExecutionPlan) -> LazyBlockList:
    """For a read-only plan, construct a LazyBlockList with ReadTasks from the
    input Datasource or Reader. Note that the plan and the underlying ReadTasks
    are not executed, only their known metadata is fetched.

    Args:
        plan: The legacy plan to execute.

    Returns:
        The output as a legacy LazyBlockList.
    """
    assert plan.is_read_only(), 'This function only supports read-only plans.'
    assert isinstance(plan._logical_plan, LogicalPlan)
    read_logical_op = plan._logical_plan.dag
    assert isinstance(read_logical_op, Read)
    ctx = DataContext.get_current()
    parallelism, _, estimated_num_blocks, k = compute_additional_split_factor(read_logical_op._datasource_or_legacy_reader, read_logical_op._parallelism, read_logical_op._mem_size, ctx.target_max_block_size, cur_additional_split_factor=None)
    read_tasks = read_logical_op._datasource_or_legacy_reader.get_read_tasks(parallelism)
    for read_task in read_tasks:
        apply_output_blocks_handling_to_read_task(read_task, k)
    block_list = LazyBlockList(read_tasks, read_logical_op.name, ray_remote_args=read_logical_op._ray_remote_args, owned_by_consumer=False)
    block_list._estimated_num_blocks = estimated_num_blocks
    return block_list