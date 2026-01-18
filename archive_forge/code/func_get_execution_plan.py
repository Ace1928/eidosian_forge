from typing import List
from ray.data._internal.logical.interfaces import (
from ray.data._internal.logical.rules._user_provided_optimizer_rules import (
from ray.data._internal.logical.rules.inherit_target_max_block_size import (
from ray.data._internal.logical.rules.operator_fusion import OperatorFusionRule
from ray.data._internal.logical.rules.randomize_blocks import ReorderRandomizeBlocksRule
from ray.data._internal.logical.rules.set_read_parallelism import SetReadParallelismRule
from ray.data._internal.logical.rules.zero_copy_map_fusion import (
from ray.data._internal.planner.planner import Planner
def get_execution_plan(logical_plan: LogicalPlan) -> PhysicalPlan:
    """Get the physical execution plan for the provided logical plan.

    This process has 3 steps:
    (1) logical optimization: optimize logical operators.
    (2) planning: convert logical to physical operators.
    (3) physical optimization: optimize physical operators.
    """
    optimized_logical_plan = LogicalOptimizer().optimize(logical_plan)
    logical_plan._dag = optimized_logical_plan.dag
    physical_plan = Planner().plan(optimized_logical_plan)
    return PhysicalOptimizer().optimize(physical_plan)