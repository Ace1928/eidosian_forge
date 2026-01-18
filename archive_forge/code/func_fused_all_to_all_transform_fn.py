from typing import List, Optional, Tuple
from ray.data._internal.compute import get_compute, is_task_compute
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.operators.actor_pool_map_operator import (
from ray.data._internal.execution.operators.base_physical_operator import (
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.operators.task_pool_map_operator import (
from ray.data._internal.logical.interfaces import PhysicalPlan, Rule
from ray.data._internal.logical.operators.all_to_all_operator import (
from ray.data._internal.logical.operators.map_operator import AbstractUDFMap
from ray.data._internal.stats import StatsDict
from ray.data.context import DataContext
def fused_all_to_all_transform_fn(blocks: List[RefBundle], ctx: TaskContext) -> Tuple[List[RefBundle], StatsDict]:
    """To fuse MapOperator->AllToAllOperator, we store the map function
            in the TaskContext so that it may be used by the downstream
            AllToAllOperator's transform function."""
    ctx.upstream_map_transformer = up_map_transformer
    ctx.upstream_map_ray_remote_args = ray_remote_args
    return down_transform_fn(blocks, ctx)