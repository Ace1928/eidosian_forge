from typing import cast, List, Optional, Sequence, Tuple
import torch
from torch.distributed._tensor._utils import compute_local_shape
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.ops.common_rules import pointwise_rule
from torch.distributed._tensor.ops.utils import (
from torch.distributed._tensor.placement_types import (
from torch.distributed.device_mesh import DeviceMesh
@register_op_strategy(aten.slice.Tensor, schema_info=RuntimeSchemaInfo(1))
def gen_slice_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    """Forward all shardings except the slice dimension."""
    defaults = (None, 0, None, None, 1)
    input_strategy, dim, start, end, step = op_schema.args_schema + defaults[len(op_schema.args_schema):]
    assert isinstance(input_strategy, OpStrategy)
    input_shape = input_strategy.output_shape
    input_ndim = input_strategy.output_ndim
    assert isinstance(dim, int)
    if start is None:
        start = 0
    if end is None or end > input_shape[dim]:
        end = input_shape[dim]
    assert isinstance(start, int)
    assert isinstance(end, int)
    assert isinstance(step, int)
    slice_dim = normalize_dim(dim, input_ndim)
    start = normalize_dim(start, input_shape[dim])
    end = normalize_dim(end, input_shape[dim])
    redundant_slice = start == 0 and end == input_shape[dim] and (step == 1)
    slice_strategy = OpStrategy([])
    for arg_strategy in input_strategy.strategies:
        arg_spec = arg_strategy.output_spec
        if not is_tensor_dim_sharded(arg_spec, dim=slice_dim) or redundant_slice:
            out_spec = DTensorSpec(mesh, arg_spec.placements)
            slice_strategy.strategies.append(PlacementStrategy(output_spec=out_spec))
    if not slice_strategy.strategies:
        for arg_strategy in input_strategy.strategies:
            arg_spec = arg_strategy.output_spec
            unshard_spec = DTensorSpec(mesh, unshard_tensor_dim(arg_spec.placements, dim=slice_dim))
            slice_strategy.strategies.append(PlacementStrategy(output_spec=unshard_spec))
    return slice_strategy