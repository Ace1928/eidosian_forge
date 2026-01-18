from typing import cast, List, Optional, Sequence, Tuple
import torch
from torch.distributed._tensor._utils import compute_local_shape
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.ops.common_rules import pointwise_rule
from torch.distributed._tensor.ops.utils import (
from torch.distributed._tensor.placement_types import (
from torch.distributed.device_mesh import DeviceMesh
@register_op_strategy([aten.equal.default, aten.is_same_size.default])
def equal_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    self_strategy, other_strategy = op_schema.args_schema
    assert isinstance(self_strategy, OpStrategy)
    assert isinstance(other_strategy, OpStrategy)
    select_strategy = self_strategy if self_strategy.max_num_shards() >= other_strategy.max_num_shards() else other_strategy
    equal_strategy = OpStrategy([])
    for arg_strategy in select_strategy.strategies:
        arg_spec = arg_strategy.output_spec
        if is_tensor_partial(arg_spec):
            output_spec = DTensorSpec(mesh=arg_spec.mesh, placements=tuple((Replicate() if isinstance(p, _Partial) else p for p in arg_spec.placements)))
            equal_strategy.strategies.append(PlacementStrategy(output_spec=output_spec))
        else:
            equal_strategy.strategies.append(PlacementStrategy(arg_spec))
    return equal_strategy