from typing import cast, List, Optional, Sequence, Tuple
import torch
from torch.distributed._tensor._utils import compute_local_shape
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.ops.common_rules import pointwise_rule
from torch.distributed._tensor.ops.utils import (
from torch.distributed._tensor.placement_types import (
from torch.distributed.device_mesh import DeviceMesh
@register_op_strategy(aten.bucketize.Tensor)
def gen_bucketize_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    """Just propagate input sharding, but expect replicated for boundaries input."""
    input_strategy = op_schema.args_schema[0]
    bucketize_strategy = OpStrategy([])
    assert isinstance(input_strategy, OpStrategy)
    for arg_strategy in input_strategy.strategies:
        arg_spec = DTensorSpec(mesh, arg_strategy.output_spec.placements)
        replica_spec = DTensorSpec(mesh, tuple([Replicate()] * mesh.ndim))
        bucketize_strategy.strategies.append(PlacementStrategy(output_spec=arg_spec, input_specs=(arg_spec, replica_spec)))
    return bucketize_strategy